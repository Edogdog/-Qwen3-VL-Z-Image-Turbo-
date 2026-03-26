"""Launch script for Style Transfer Gradio demo"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path

# Check for VLM model (in parent directory of workspace)
# lab5-style-transfer -> modelscope-workshop -> modelscope-contest
workspace_dir = Path(__file__).resolve().parent
project_dir = workspace_dir.parent.parent  # Go up two levels: lab5 -> workshop -> contest
vlm_model_dir = project_dir / 'Qwen3-VL-4B-Instruct-int4-ov'
if not vlm_model_dir.exists():
    vlm_model_dir = workspace_dir / 'Qwen3-VL-4B-Instruct-int4-ov'
if not vlm_model_dir.exists():
    vlm_model_dir = Path('Qwen3-VL-4B-Instruct-int4-ov')
if not vlm_model_dir.exists():
    print("ERROR: VLM model not found at 'Qwen3-VL-4B-Instruct-int4-ov'")
    print("Please download it first using ModelScope:")
    print("  from modelscope import snapshot_download")
    print("  snapshot_download('snake7gun/Qwen3-VL-4B-Instruct-int4-ov', local_dir='Qwen3-VL-4B-Instruct-int4-ov')")
    sys.exit(1)

# Patch VLM config if needed
with open(vlm_model_dir / 'config.json') as f:
    config = json.load(f)
if 'vision_config' in config and 'embed_dim' not in config['vision_config']:
    config['vision_config']['embed_dim'] = config['vision_config']['hidden_size']
    with open(vlm_model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Patched VLM config.json to add embed_dim")

# Check for Image model (in parent directory of workspace)
image_model_dir = project_dir / 'Z-Image-Turbo-int4-ov'
if not image_model_dir.exists():
    image_model_dir = workspace_dir / 'Z-Image-Turbo-int4-ov'
if not image_model_dir.exists():
    image_model_dir = Path('Z-Image-Turbo-int4-ov')
if not image_model_dir.exists():
    print("ERROR: Image model not found at 'Z-Image-Turbo-int4-ov'")
    print("Please download it first using ModelScope:")
    print("  from modelscope import snapshot_download")
    print("  snapshot_download('ZhipuAI/ImageGen-Turbo-Instruct-int4-ov', local_dir='Z-Image-Turbo-int4-ov')")
    sys.exit(1)

print("All models found. Loading...")

# Import after path setup
from gradio_helper import make_demo, ZImageTurboOV
from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVQwen2VLForCausalLM
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

# IMPORTANT: Monkey patch BEFORE loading VLM
MODEL_TYPE_TO_CLS_MAPPING['qwen3_vl'] = _OVQwen2VLForCausalLM

# Load VLM model
print("Loading Qwen3-VL model...")
vlm_model = OVModelForVisualCausalLM.from_pretrained(vlm_model_dir, device='CPU')
vlm_processor = AutoProcessor.from_pretrained(
    vlm_model_dir,
    min_pixels=256*28*28,
    max_pixels=1280*28*28,
    fix_mistral_regex=True
)

# Load Image generation model
print("Loading Z-Image-Turbo model...")
image_generator = ZImageTurboOV(str(image_model_dir), device='CPU')

# Launch demo
print("Launching Gradio demo...")
demo = make_demo(vlm_model, vlm_processor, image_generator, model_name='Style Transfer')
demo.launch(debug=True, share=True, server_name="0.0.0.0")
