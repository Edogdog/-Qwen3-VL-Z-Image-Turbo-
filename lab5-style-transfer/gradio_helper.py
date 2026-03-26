import re
import gradio as gr
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from z_image_turbo_ov import ZImageTurboOV
import numpy as np


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _remove_image_special(text):
    text = text.replace("<ref>", "").replace("</ref>", "")
    return re.sub(r"<box>.*?(</box>|$)", "", text)


def transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message["content"]:
            if "image" in item:
                new_item = {"type": "image", "image": item["image"]}
            elif "text" in item:
                new_item = {"type": "text", "text": item["text"]}
            elif "video" in item:
                new_item = {"type": "video", "video": item["video"]}
            else:
                continue
            new_content.append(new_item)

        new_message = {"role": message["role"], "content": new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def analyze_style(vlm_model, vlm_processor, style_image_path, progress=gr.Progress(track_tqdm=True)):
    """使用 VLM 分析风格参考图"""
    style_prompt = """请分析这张图片的艺术风格特征，输出结构化的风格描述，包括：
1. 色彩风格：主要色调、色彩饱和度、对比度
2. 氛围情感：整体情绪、场景氛围
3. 构图特点：画面布局、视角、景别
4. 艺术风格：流派、表现手法、独特元素
5. 视觉元素：重要物体、纹理、质感

请用简洁的中文描述输出，以便用于图像生成参考。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": style_image_path},
                {"type": "text", "text": style_prompt},
            ]
        }
    ]

    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    # Ensure inputs are on the correct device
    inputs = {k: v.to(vlm_model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    tokenizer = vlm_processor.tokenizer
    streamer = TextIteratorStreamer(tokenizer, timeout=300.0, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {"max_new_tokens": 300, "streamer": streamer, **inputs}

    thread = Thread(target=vlm_model.generate, kwargs=gen_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield _remove_image_special(generated_text)

    return _remove_image_special(generated_text)


def make_demo(vlm_model, vlm_processor, image_generator, model_name="Style Transfer"):
    """创建风格迁移 Gradio 演示界面"""

    def analyze_style_wrapper(style_image, progress=gr.Progress(track_tqdm=True)):
        """包装风格分析函数"""
        if style_image is None:
            return "", "Please upload a style reference image"

        # Handle Gradio image dict or file path
        if isinstance(style_image, dict):
            filepath = style_image.get("path", "")
        elif isinstance(style_image, str):
            filepath = style_image
        else:
            filepath = getattr(style_image, 'name', '')
        result = ""
        for text in analyze_style(vlm_model, vlm_processor, filepath, progress):
            result = text
        if result:
            return result, result
        else:
            return "Failed to analyze style", ""

    def generate_wrapper(content_prompt, style_description, progress=gr.Progress(track_tqdm=True)):
        """包装图像生成函数"""
        if not content_prompt or not content_prompt.strip():
            return None, "Please input content description"

        if style_description and style_description.strip():
            final_prompt = f"{content_prompt}, {style_description}"
        else:
            final_prompt = content_prompt

        progress(0.1, desc="Generating image...")
        try:
            image = image_generator.generate(
                prompt=final_prompt,
                height=512,
                width=512,
                num_inference_steps=100,
                seed=int(np.random.randint(0, 1000000))
            )
            return image, "Image generated successfully"
        except Exception as e:
            return None, f"Generation failed: {str(e)}"

    with gr.Blocks() as demo:
        gr.Markdown(f"""# Image Style Transfer - {model_name} OpenVINO""")
        gr.Markdown("""**Usage**:
1. Upload a style reference image on the left
2. The system will automatically analyze the style features
3. Enter content description
4. Click "Generate Image" to create stylized image
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Style Analysis")
                style_image_input = gr.Image(type="filepath", label="Upload Style Reference")
                analyze_btn = gr.Button("Analyze Style", variant="primary")

                gr.Markdown("### Content Description")
                content_prompt_input = gr.Textbox(
                    lines=3,
                    label="Enter content description",
                    placeholder="e.g., An elegant Chinese classical beauty..."
                )
                generate_btn = gr.Button("Generate Image", variant="primary")

            with gr.Column(scale=1):
                style_description_output = gr.Textbox(
                    lines=5,
                    label="Style Analysis Result",
                    interactive=True
                )
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Textbox(label="Generation Info", interactive=False)

        # Style analysis event
        analyze_btn.click(
            analyze_style_wrapper,
            inputs=[style_image_input],
            outputs=[style_description_output],
            show_progress="minimal"
        )

        # Image generation event
        generate_btn.click(
            generate_wrapper,
            inputs=[content_prompt_input, style_description_output],
            outputs=[output_image, generation_info],
            show_progress="minimal"
        )

        gr.Markdown("""
        ### Tips
        - Style analysis results are automatically filled into generation
        - You can also manually edit the style description
        - Different style reference images produce different effects
        """)

    return demo
