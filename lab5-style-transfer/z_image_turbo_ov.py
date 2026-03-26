"""
完整的 Z-Image-Turbo OpenVINO 推理实现
支持从文本提示词生成图像
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List
import openvino as ov
from transformers import AutoTokenizer


class ZImageTurboOV:
    """Z-Image-Turbo OpenVINO 推理类"""

    def __init__(self, model_dir: str, device: str = 'CPU'):
        self.model_dir = Path(model_dir)
        self.device = device
        self.core = ov.Core()

        print("Loading Z-Image-Turbo models...")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / 'tokenizer')

        # 加载并编译文本编码器
        self.text_encoder = self.core.compile_model(
            self.core.read_model(self.model_dir / 'text_encoder' / 'openvino_model.xml'),
            device
        )

        # 加载并编译 transformer
        self.transformer = self.core.compile_model(
            self.core.read_model(self.model_dir / 'transformer' / 'openvino_model.xml'),
            device
        )

        # 加载并编译 VAE 解码器
        self.vae_decoder = self.core.compile_model(
            self.core.read_model(self.model_dir / 'vae_decoder' / 'openvino_model.xml'),
            device
        )

        # 加载 scheduler 配置
        import json
        scheduler_config = json.load(open(self.model_dir / 'scheduler' / 'scheduler_config.json'))
        self.num_train_timesteps = scheduler_config.get('num_train_timesteps', 1000)
        self.shift = scheduler_config.get('shift', 3.0)

        # 加载 VAE 配置 (用于缩放因子)
        vae_config = json.load(open(self.model_dir / 'vae_decoder' / 'config.json'))
        self.vae_scaling_factor = vae_config.get('scaling_factor', 0.3611)
        self.vae_shift_factor = vae_config.get('shift_factor', 0.1159)

        print("Z-Image-Turbo models loaded successfully!")

    def encode_prompt(self, prompt: Union[str, List[str]]) -> np.ndarray:
        """将文本提示词编码为 embeddings"""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="np"
        )

        input_ids = tokens.input_ids.astype(np.int64)
        attention_mask = tokens.attention_mask.astype(np.int64)

        # 通过文本编码器
        text_req = self.text_encoder.create_infer_request()
        text_req.set_input_tensor(0, ov.Tensor(input_ids))
        text_req.set_input_tensor(1, ov.Tensor(attention_mask))
        text_req.infer()

        text_embeds = text_req.get_output_tensor(0).data
        return text_embeds  # [batch, seq_len, hidden_dim]

    @staticmethod
    def get_scalings(sigma: np.ndarray, shift: float = 3.0) -> tuple:
        """计算 Flow Matching 的缩放因子"""
        sigma = np.clip(sigma, 1e-6, None)  # Avoid division by zero
        c_skip = (sigma ** 2 + shift * 2) / (2 * sigma ** 2)
        c_out = sigma / np.sqrt(sigma ** 2 + 1)
        c_in = 1 / np.sqrt(sigma ** 2 + 1)
        return c_skip, c_out, c_in

    def generate(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        生成图像

        Args:
            prompt: 文本提示词
            height: 输出高度
            width: 输出宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导_scale (Z-Image-Turbo 通常为 0)
            seed: 随机种子

        Returns:
            PIL.Image: 生成的图像
        """
        from PIL import Image

        # 计算 latent 尺寸 (8x 下采样)
        latent_height = height // 8
        latent_width = width // 8

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)

        # 编码 prompt
        text_embeds = self.encode_prompt(prompt)  # [1, seq_len, hidden_dim]

        # 初始化纯噪声 latent (5D: [batch, channels, time, height, width])
        # Z-Image-Turbo 使用 S3-DiT，latent 是 5D 的
        noise = np.random.randn(1, 16, 1, latent_height, latent_width).astype(np.float32)
        latents = noise.copy()

        # Flow Matching 推理
        # 使用离散时间步
        timesteps = np.linspace(1, 0, num_inference_steps, dtype=np.float32)

        print(f"Running {num_inference_steps} inference steps...")

        for i, t in enumerate(timesteps):
            # 转换为 sigma (时间缩放)
            sigma = t * self.shift

            # 缩放输入
            scaled_latents = latents / np.sqrt(sigma ** 2 + 1)

            # Transformer 推理
            transformer_req = self.transformer.create_infer_request()
            transformer_req.set_input_tensor(0, ov.Tensor(scaled_latents))
            transformer_req.set_input_tensor(1, ov.Tensor(np.array([sigma], dtype=np.float32)))
            transformer_req.set_input_tensor(2, ov.Tensor(text_embeds.astype(np.float32)))
            transformer_req.infer()

            model_output = transformer_req.get_output_tensor(0).data  # [16, 1, H, W]
            # 转换为 [1, 16, 1, H, W]
            model_output = model_output.reshape(1, 16, 1, latent_height, latent_width)

            # Flow Matching 更新
            # 简化的更新：latents = latents - sigma * model_output
            latents = latents - sigma * model_output

            print(f"  Step {i+1}/{num_inference_steps}, t={t:.3f}, sigma={sigma:.3f}")

        # VAE 解码
        # VAE 期望 4D latent: [batch, channels, height, width]
        # 使用 VAE 配置中的缩放因子进行缩放
        latents_for_vae = latents.squeeze(2)  # [1, 16, 1, H, W] -> [1, 16, H, W]
        scaled_latents_for_vae = (latents_for_vae - self.vae_shift_factor) / self.vae_scaling_factor

        vae_req = self.vae_decoder.create_infer_request()
        vae_req.set_input_tensor(0, ov.Tensor(scaled_latents_for_vae))
        vae_req.infer()

        decoded = vae_req.get_output_tensor(0).data  # [1, 3, H*8, W*8]

        # 转换为图像
        # 反归一化 (假设 VAE 输出在 [-1, 1] 范围)
        image_array = np.clip((decoded.squeeze(0).transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
        image_array = (image_array * 255).astype(np.uint8)

        return Image.fromarray(image_array)


def main():
    """测试 Z-Image-Turbo 生成"""
    import os

    model_dir = "Z-Image-Turbo-int4-ov"

    if not Path(model_dir).exists():
        print(f"Model not found at {model_dir}")
        print("Please download the model first using ModelScope")
        return

    print("Initializing Z-Image-Turbo...")
    generator = ZImageTurboOV(model_dir, device='CPU')

    prompt = "A beautiful Chinese classical beauty in traditional Hanfu, elegant posture"
    print(f"\nGenerating image with prompt: {prompt}")

    image = generator.generate(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=9,
        guidance_scale=0.0,
        seed=42
    )

    output_path = "generated_image.png"
    image.save(output_path)
    print(f"\nImage saved to: {output_path}")

    return image


if __name__ == "__main__":
    main()
