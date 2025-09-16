import torch
from diffusers import StableDiffusionXLPipeline
import gradio as gr
from huggingface_hub import login

# Login to Hugging Face
login(token="hf_token_should_be_here")

# Load SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Enable memory-efficient attention
pipe.enable_xformers_memory_efficient_attention()

# Function to generate image
def generate_image(prompt, negative_prompt="lowres, blurry, deformed, bad anatomy", guidance_scale=12.0, steps=50):
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]
    return img

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your creative prompt here..."),
    outputs=gr.Image(type="pil"),
    title="Kapil Anandh's SDXL Image Generator",
    description="Learning session project: Generate stunning photorealistic images with Stable Diffusion XL (SDXL)."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
