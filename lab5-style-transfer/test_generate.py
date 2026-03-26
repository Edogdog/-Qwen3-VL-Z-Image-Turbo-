"""Quick test for image generation only"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
import time
import numpy as np

# Setup paths
workspace_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path('.')
project_dir = workspace_dir.parent.parent
image_model_dir = project_dir / 'Z-Image-Turbo-int4-ov'

print(f"Image model path: {image_model_dir}")
print(f"Exists: {image_model_dir.exists()}")

if not image_model_dir.exists():
    print("ERROR: Image model not found!")
    sys.exit(1)

print("\nLoading Z-Image-Turbo model...")
from gradio_helper import ZImageTurboOV

start = time.time()
generator = ZImageTurboOV(str(image_model_dir), device='CPU')
print(f"Model loaded in {time.time()-start:.1f}s")

# Test encode_prompt first
print("\n[1/4] Testing encode_prompt...")
start = time.time()
text_embeds = generator.encode_prompt("A beautiful sunset")
print(f"encode_prompt took {time.time()-start:.1f}s, output shape: {text_embeds.shape}")
print(f"Expected Input 2 shape: [?,?,2560], actual: {text_embeds.shape}")

# Test with a single step to see if inference works
print("\n[2/4] Testing single inference step...")
import openvino as ov

# Check model input shapes
print("Transformer input shapes:")
for i, inp in enumerate(generator.transformer.inputs):
    print(f"  Input {i}: {inp.get_partial_shape()}")

latent_height, latent_width = 64, 64
noise = np.random.randn(1, 16, 1, latent_height, latent_width).astype(np.float32)
latents = noise.copy()
sigma = np.array([3.0], dtype=np.float32)  # array with one element
scaled_latents = latents / np.sqrt(sigma ** 2 + 1)

start = time.time()
transformer_req = generator.transformer.create_infer_request()
transformer_req.set_input_tensor(0, ov.Tensor(scaled_latents))
transformer_req.set_input_tensor(1, ov.Tensor(sigma))  # array [3.0]
transformer_req.set_input_tensor(2, ov.Tensor(text_embeds.astype(np.float32)))
transformer_req.infer()
print(f"Single transformer step took {time.time()-start:.1f}s")

# Check VAE input shape
print("\nVAE input shapes:")
for i, inp in enumerate(generator.vae_decoder.inputs):
    print(f"  Input {i}: {inp.get_partial_shape()}")

print("\n[3/4] Full generation (9 steps)...")
start = time.time()
image = generator.generate_image(
    prompt='A beautiful sunset over mountains',
    height=512,
    width=512,
    num_inference_steps=5,
    seed=42
)
print(f"Full generation took {time.time()-start:.1f}s")

output_path = 'test_output.png'
image.save(output_path)
print(f"\nImage saved to: {output_path}")
print("SUCCESS!")
