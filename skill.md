---
name: style-transfer
description: 基于 Qwen3-VL 视觉语言模型分析风格参考图，结合 Z-Image-Turbo 图像生成模型创建风格化图像。VLM 提取语义级风格特征，实现零样本风格迁移，支持在 Intel AI PC 上端侧部署运行。
---

# Style Transfer Skill for OpenClaw

> 图像风格迁移 Skill - 基于 Qwen3-VL + Z-Image-Turbo + OpenVINO

## Skill 基本信息

**版本**: 1.0.0
**作者**: snake7gun
**标签**: image-generation, style-transfer, openvino, qwen3-vl, z-image-turbo, intel-ai-pc

---

## 功能说明

### 核心功能

1. **风格分析**: 使用 Qwen3-VL-4B-Instruct-int4 分析用户上传的风格参考图，提取结构化风格描述
2. **智能生成**: 结合风格描述和内容提示词，通过 Z-Image-Turbo 生成新图像
3. **端侧部署**: 基于 OpenVINO，在 Intel AI PC (CPU) 上高效运行，无需 GPU

### 使用场景

- 艺术风格迁移（照片转油画、水墨画、插画风格）
- 品牌视觉统一（产品图调整为品牌风格）
- 教育内容创作（为教学材料添加特定艺术风格）
- 游戏素材生成（创建特定风格的角色或场景）

---

## 输入输出

### 输入 (Input)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| style_image | ImageData | 否 | 风格参考图（支持 JPG/PNG），不传则使用默认风格 |
| content_prompt | string | 是 | 内容描述文本 |

```json
{
  "style_image": {
    "path": "/path/to/style.jpg",
    "url": null,
    "meta": {"_type": "gradio.FileData"}
  },
  "content_prompt": "一位优雅的中国古典美女"
}
```

### 输出 (Output)

| 字段 | 类型 | 说明 |
|------|------|------|
| generated_image | ImageData | 生成的风格化图像（512x512 PNG） |
| style_description | string | VLM 分析的风格描述 |
| generation_info | string | 生成状态信息 |

```json
{
  "generated_image": {
    "path": "/path/to/output.png",
    "url": null
  },
  "style_description": "古典东方美学风格，浓郁的中国传统色彩（红、金、黑），工笔画细腻线条...",
  "generation_info": "Image generated successfully"
}
```

---

## 技术实现

### 模型架构

| 模型 | 类型 | 说明 |
|------|------|------|
| Qwen3-VL-4B-Instruct-int4-ov | VLM | 风格分析与理解 |
| Z-Image-Turbo-Instruct-int4-ov | 生成模型 | Flow Matching 图像生成 |

### 处理流程

```
风格参考图 → Qwen3-VL → 风格描述
                               ↓
内容描述 → 拼接 Prompt → Z-Image-Turbo → 风格化图像
```

### 性能指标

| 指标 | 数值 |
|------|------|
| 风格分析 | ~30-60 秒 (CPU) |
| 图像生成 (100步) | ~10-15 分钟 (CPU) |
| 输出分辨率 | 512x512 |
| 内存占用 | ~4GB (双模型) |

> 注：实际推理步数由 `num_inference_steps` 参数控制，9步约 1-2 分钟（快速预览），100步约 10-15 分钟（高质量输出）

---

## 配置参数

### 可调参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_inference_steps | int | 100 | 推理步数，越多越精细（9步快速预览，100步高质量输出） |
| seed | int | 随机 | 随机种子，用于复现 |

### 模型路径

模型默认路径：
- VLM: `../Qwen3-VL-4B-Instruct-int4-ov`
- 生成模型: `../Z-Image-Turbo-int4-ov`

（相对于 `lab5-style-transfer` 目录）

---

## 依赖项

```yaml
python_version: ">=3.10"
packages:
  - openvino>=2025.4
  - gradio>=6.0
  - optimum-intel
  - transformers
  - qwen-vl-utils
  - pillow
  - numpy
```

---

## 错误处理

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| E001 | 无效的图像格式 | 上传 JPG/PNG 格式的图像 |
| E002 | 内容描述为空 | 请提供有效的内容描述 |
| E003 | 模型加载失败 | 检查模型路径是否正确 |
| E004 | 生成超时 | 减少推理步数（num_inference_steps） |
| E005 | 图像呈马赛克 | 检查 VAE 缩放因子配置 |

---

## 项目结构

```
modelscope-workshop/
└── lab5-style-transfer/
    ├── gradio_helper.py       # Gradio UI 界面与事件绑定
    ├── z_image_turbo_ov.py    # Z-Image-Turbo 推理引擎（核心类 ZImageTurboOV）
    ├── launch_demo.py         # 启动脚本：模型加载与组装
    ├── test_generate.py       # 测试脚本
    └── notebook_utils.py      # 工具函数

modelscope-contest/            # 项目根目录
├── Qwen3-VL-4B-Instruct-int4-ov/   # VLM 模型（风格分析）
├── Z-Image-Turbo-int4-ov/          # 图像生成模型
└── modelscope-workshop/
    └── lab5-style-transfer/        # 应用代码
```

---

## 环境要求

### 硬件环境

| 项目 | 要求 |
|------|------|
| 设备 | Intel AI PC / PC（支持 OpenVINO） |
| 内存 | ≥ 8GB RAM |
| 存储 | ≥ 10GB 可用空间 |

### 软件环境

| 项目 | 版本要求 |
|------|----------|
| Python | ≥ 3.10 |
| openvino | ≥ 2025.4 |
| torch | 2.8 (CPU) |
| gradio | ≥ 6.0 |

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/openvinotoolkit/modelscope-workshop.git
cd modelscope-workshop

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型（通过 ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('snake7gun/Qwen3-VL-4B-Instruct-int4-ov', local_dir='../Qwen3-VL-4B-Instruct-int4-ov')"
python -c "from modelscope import snapshot_download; snapshot_download('ZhipuAI/ImageGen-Turbo-Instruct-int4-ov', local_dir='../Z-Image-Turbo-int4-ov')"
```

---

## 模型说明

### VLM 模型 (Qwen3-VL-4B-Instruct-int4-ov)

| 项目 | 说明 |
|------|------|
| 模型ID | `snake7gun/Qwen3-VL-4B-Instruct-int4-ov` |
| 参数量 | 4B (INT4 量化) |
| 功能 | 风格分析 - 提取图像的艺术风格特征 |
| 输入 | 风格参考图 |
| 输出 | 结构化风格描述文本 |

### 图像生成模型 (Z-Image-Turbo-Instruct-int4-ov)

| 项目 | 说明 |
|------|------|
| 模型ID | `ZhipuAI/ImageGen-Turbo-Instruct-int4-ov` |
| 架构 | Flow Matching + VAE 解码器 |
| 功能 | 文本到图像生成 |
| 输入 | 文本描述（内容 + 风格） |
| 输出 | 512x512 RGB 图像 |

### 模型目录结构

```
Qwen3-VL-4B-Instruct-int4-ov/
├── config.json              # 模型配置
├── model.safetensors        # 模型权重
├── tokenizer/               # Tokenizer
│   └── ...
└── processor/               # 处理器配置

Z-Image-Turbo-int4-ov/
├── text_encoder/            # 文本编码器
│   └── openvino_model.xml
├── transformer/             # Transformer（Flow Matching 主干）
│   └── openvino_model.xml
├── vae_decoder/             # VAE 解码器
│   ├── openvino_model.xml
│   └── config.json
└── scheduler/               # 调度器配置
    └── scheduler_config.json
```

---

## 相关代码

### 核心类：ZImageTurboOV

```python
# z_image_turbo_ov.py
class ZImageTurboOV:
    """Z-Image-Turbo OpenVINO 推理类"""

    def __init__(self, model_dir, device='CPU'):
        # 加载 text_encoder, transformer, vae_decoder
        # 加载 tokenizer 和 scheduler 配置

    def encode_prompt(self, prompt):
        """将文本提示词编码为 embeddings"""

    def generate(self, prompt, height=512, width=512,
                 num_inference_steps=100, seed=None):
        """生成图像 - Flow Matching 推理流程"""
```

### 风格分析函数

```python
# gradio_helper.py
def analyze_style(vlm_model, vlm_processor, style_image_path):
    """使用 Qwen3-VL 分析风格参考图，返回结构化风格描述"""
    # 输入：风格参考图路径
    # 输出：流式风格描述文本
```

### 启动入口

```python
# launch_demo.py
from gradio_helper import make_demo, ZImageTurboOV
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

# 加载模型
vlm_model = OVModelForVisualCausalLM.from_pretrained(vlm_model_dir)
vlm_processor = AutoProcessor.from_pretrained(vlm_model_dir)
image_generator = ZImageTurboOV(image_model_dir)

# 启动 Gradio
demo = make_demo(vlm_model, vlm_processor, image_generator)
demo.launch()
```

### 调用流程

```
用户上传风格图 → analyze_style() → Qwen3-VL → 风格描述
                                    ↓
用户输入内容描述 + 风格描述 → image_generator.generate()
                                    ↓
                            Z-Image-Turbo → 风格化图像
```

---

## 使用限制

- 单次生成的图像尺寸: 512x512
- 单个文件大小限制: 10MB
- 支持的图像格式: JPEG, PNG

---

## 许可证

Apache 2.0

---
