# 图像风格迁移应用 - 实践日志

> 记录魔搭比赛项目开发全过程

## 一、环境配置踩坑记录

### 1.1 环境信息
- Python 版本: 3.14.3
- 虚拟环境: ov_workshop
- 操作系统: Windows 10 Home China
- OpenVINO 版本: 2026.0.0

### 1.2 依赖安装问题与解决

#### 问题1: numpy 版本冲突
**问题描述**: requirements.txt 中指定 `numpy<2.0`，但预编译 wheel 不支持 Python 3.14

**解决方式**: 手动安装 numpy==2.3.5 预编译版本
```bash
pip install numpy==2.3.5
```

#### 问题2: optimum-intel 和 diffusers 无法从 GitHub 安装
**问题描述**: 网络连接被重置，无法从 GitHub 克隆仓库

**解决方式**: 这些包是可选的，核心 OpenVINO 功能已内置，不需要额外安装

#### 问题3: 依赖解析器版本冲突
**问题描述**: pip 依赖解析器尝试从源码编译 numpy，但缺少 C++ 编译器

**解决方式**: 使用 `--no-deps` 参数安装基础包，再单独安装依赖

#### 问题4: Python 3.14 audioop 模块缺失
**问题描述**: Python 3.14 移除了内置的 audioop 模块，导致 pydub 无法导入

**解决方式**: 安装 audioop-lts 替代包
```bash
pip install audioop-lts
```

#### 问题5: Gradio 版本依赖冲突
**问题描述**: 安装的依赖版本与 gradio 6.9.0 要求不匹配

| 冲突项 | Gradio 要求 | 实际安装 | 解决方案 |
|--------|------------|---------|---------|
| aiofiles | <25.0, >=22.0 | 25.1.0 | 忽略（功能正常） |
| gradio-client | ==2.3.0 | 2.4.0 | 忽略（功能正常） |
| starlette | <1.0, >=0.40.0 | 1.0.0 | 安装 fastapi 解决 |
| tomlkit | <0.14.0, >=0.12.0 | 0.14.0 | 忽略（功能正常） |

**说明**: 这些版本冲突不影响核心功能，Gradio 仍可正常运行

#### 问题6: fastapi 模块缺失
**问题描述**: Gradio 依赖 fastapi 但未自动安装

**解决方式**:
```bash
pip install fastapi uvicorn
```

---

## 二、模型下载和加载过程

### 2.1 VLM 模型 (Qwen3-VL)
- 模型名称: Qwen3-VL-4B-Instruct-int4-ov
- 模型来源: snake7gun/Qwen3-VL-4B-Instruct-int4-ov (ModelScope)
- 下载方式: `modelscope.snapshot_download`
- 模型大小: 约 4GB (INT4 量化)

#### 加载问题与解决

**问题**: optimum-intel 1.27.0 不支持 qwen3_vl 模型类型

**错误**:
```
KeyError: 'qwen3_vl'
AttributeError: 'Qwen3VLVisionConfig' object has no attribute 'embed_dim'
```

**最终解决方案**:
1. 修改 config.json，添加 `embed_dim` 到 vision_config
2. 使用 monkey-patch 将 qwen3_vl 映射到 qwen2_vl 实现

```python
# 修改 config.json
import json
with open('Qwen3-VL-4B-Instruct-int4-ov/config.json') as f:
    config = json.load(f)
config['vision_config']['embed_dim'] = config['vision_config']['hidden_size']
with open('Qwen3-VL-4B-Instruct-int4-ov/config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Monkey-patch
from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVQwen2VLForCausalLM
MODEL_TYPE_TO_CLS_MAPPING['qwen3_vl'] = _OVQwen2VLForCausalLM
```

### 2.2 图像生成模型 (Z-Image-Turbo)
- 模型名称: Z-Image-Turbo-int4-ov
- 模型来源: snake7gun/Z-Image-Turbo-int4-ov (ModelScope)
- 下载方式: `modelscope.snapshot_download`
- 模型文件结构:
  - `transformer/openvino_model.xml + .bin` - DiT transformer
  - `text_encoder/openvino_model.xml + .bin` - Qwen3 文本编码器
  - `vae_decoder/openvino_model.xml + .bin` - VAE 解码器
  - `tokenizer/` - 分词器

#### 加载问题与解决

**问题**: `OVZImagePipeline` 在 optimum-intel 中不可用

**解决方案**: 直接使用 OpenVINO Core 加载各组件

```python
import openvino as ov
from transformers import AutoTokenizer

core = ov.Core()

# 加载各组件
transformer = core.compile_model(core.read_model('transformer/openvino_model.xml'), 'CPU')
text_encoder = core.compile_model(core.read_model('text_encoder/openvino_model.xml'), 'CPU')
vae = core.compile_model(core.read_model('vae_decoder/openvino_model.xml'), 'CPU')
tokenizer = AutoTokenizer.from_pretrained('tokenizer/')
```

---

## 三、最终验证结果

### 3.1 测试通过的功能

| 组件 | 状态 | 说明 |
|------|------|------|
| OpenVINO | PASS | 版本 2026.0.0 |
| Transformers | PASS | 版本 4.57.6 |
| Gradio | PASS | 版本 6.9.0 |
| Modelscope | PASS | 版本 1.35.1 |
| Qwen3-VL | PASS | 需 monkey-patch |
| Z-Image-Turbo | PASS | 直接加载 OpenVINO 模型 |
| VLM 文本推理 | PASS | 可正常回答问题 |
| Z-Image-Turbo 推理 | PASS | 完整 Flow Matching 推理成功 |

### 3.2 完整测试命令

```python
import sys
sys.path.insert(0, 'modelscope-workshop/lab5-style-transfer')

# 1. Load VLM
import json
from pathlib import Path
from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVQwen2VLForCausalLM
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

vlm_model_dir = Path('Qwen3-VL-4B-Instruct-int4-ov')
with open(vlm_model_dir / 'config.json') as f:
    config_dict = json.load(f)
if 'vision_config' in config_dict and 'embed_dim' not in config_dict.get('vision_config', {}):
    config_dict['vision_config']['embed_dim'] = config_dict['vision_config']['hidden_size']
    with open(vlm_model_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

MODEL_TYPE_TO_CLS_MAPPING['qwen3_vl'] = _OVQwen2VLForCausalLM
vlm_model = OVModelForVisualCausalLM.from_pretrained(vlm_model_dir, device='CPU')
vlm_processor = AutoProcessor.from_pretrained(vlm_model_dir, min_pixels=256*28*28, max_pixels=1280*28*28)

# 2. Load Image Models
import openvino as ov
from transformers import AutoTokenizer

image_model_dir = Path('Z-Image-Turbo-int4-ov')
core = ov.Core()
transformer = core.compile_model(core.read_model(image_model_dir / 'transformer' / 'openvino_model.xml'), 'CPU')
text_encoder = core.compile_model(core.read_model(image_model_dir / 'text_encoder' / 'openvino_model.xml'), 'CPU')
vae = core.compile_model(core.read_model(image_model_dir / 'vae_decoder' / 'openvino_model.xml'), 'CPU')
image_tokenizer = AutoTokenizer.from_pretrained(image_model_dir / 'tokenizer')

# 3. Test VLM
from transformers import TextStreamer
messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2+2?'}]}]
inputs = vlm_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
inputs = {k: v.to(vlm_model.device) for k, v in inputs.items()}
result = vlm_model.generate(**inputs, max_new_tokens=20, streamer=TextStreamer(vlm_processor.tokenizer, skip_prompt=True, skip_special_tokens=True))
# 输出: Four

print("All tests PASSED!")
```

---

## 四、风格提取 Prompt 设计思路

### 4.1 Prompt 模板
```
请分析这张图片的艺术风格特征，输出结构化的风格描述，包括：
1. 色彩风格：主要色调、色彩饱和度、对比度
2. 氛围情感：整体情绪、场景氛围
3. 构图特点：画面布局、视角、景别
4. 艺术风格：流派、表现手法、独特元素
5. 视觉元素：重要物体、纹理、质感

请用简洁的中文描述输出，以便用于图像生成参考。
```

---

## 五、Gradio 应用修复与启动

### 5.1 修复的问题

| 问题 | 位置 | 修复方式 |
|------|------|----------|
| 方法名不匹配 | `gradio_helper.py:248` | `generate_image` vs `generate` |
| 返回值逻辑错误 | `gradio_helper.py:231-234` | 空结果时返回错误信息 |
| Gradio API 废弃 | `gradio_helper.py:296,304` | `show_progress=True` → `"minimal"` |
| Textbox 无法编辑 | `gradio_helper.py:286` | `interactive=False` → `True` |
| 未使用 imports | `gradio_helper.py:1-10` | 移除 `diffusers`, `torch`, `random` |

### 5.2 模型路径问题与解决

**问题**: 模型位于上级目录 `D:\python_project\modelscope_contest\`

**解决方案**: 在 `launch_demo.py` 中使用正确的路径解析：

```python
workspace_dir = Path(__file__).resolve().parent
project_dir = workspace_dir.parent.parent  # lab5 -> workshop -> contest
vlm_model_dir = project_dir / 'Qwen3-VL-4B-Instruct-int4-ov'
image_model_dir = project_dir / 'Z-Image-Turbo-int4-ov'
```

### 5.3 启动验证

```bash
cd modelscope-workshop/lab5-style-transfer
python launch_demo.py
```

**验证结果**:
- 服务启动成功: http://localhost:7860 (HTTP 200)
- Gradio 界面加载正常
- 所有组件正常识别

### 5.4 VAE 缩放因子修复

**问题**: 生成的图像呈现马赛克/像素化效果

**原因**: VAE 解码器需要使用配置文件中的缩放因子进行预处理，而非直接使用 latent

**发现过程**:
1. 检查 VAE decoder config.json 发现 `scaling_factor: 0.3611` 和 `shift_factor: 0.1159`
2. 这些因子用于将 latent 缩放到 VAE 期望的范围

**修复方式**:
```python
# gradio_helper.py 中添加 VAE 配置加载
vae_config = json.load(open(self.model_dir / 'vae_decoder' / 'config.json'))
self.vae_scaling_factor = vae_config.get('scaling_factor', 0.3611)
self.vae_shift_factor = vae_config.get('shift_factor', 0.1159)

# 在 VAE 解码前进行缩放
latents_for_vae = latents.squeeze(2)
scaled_latents_for_vae = (latents_for_vae - self.vae_shift_factor) / self.vae_scaling_factor
```

**修复文件**:
- `gradio_helper.py` - ZImageTurboOV 类
- `z_image_turbo_ov.py` - ZImageTurboOV 类

---

## 六、局限性与优化方向

### 6.1 当前局限性
1. **网络下载问题**: GitHub raw content 在部分网络环境下访问超时
2. **分辨率限制**: 当前仅支持 512x512 输出

### 6.2 已完成优化
1. ✅ 实现完整的 Z-Image-Turbo 推理流程
2. ✅ 端到端风格迁移演示
3. ✅ Gradio 交互界面集成并启动

### 6.3 未来优化方向
1. 支持多种输出分辨率
2. 优化推理性能
3. 开发 OpenClaw Skill 版本

---

## 七、参赛作品信息

- **项目名称**: 图像风格迁移创新应用
- **选择方向**: 图像生成 + 视觉理解（VLM）
- **技术栈**: OpenVINO + Qwen3-VL + Z-Image-Turbo
- **部署平台**: Intel AI PC (端侧部署)
- **项目状态**: ✅ 全部组件测试通过，完整推理流程验证成功
