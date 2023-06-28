import torch
import os
from datetime import datetime

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

#####################
#   CONFIGURATION   #
#####################

model_id = "models/your_custom_model"

seed = 3
prompt = ""
negative_prompt = ""
image_size = 768
num_inference_steps = 25
num_images = 10

#####################

load_sampler = lambda config: DPMSolverMultistepScheduler.from_config(
    config, use_karras=True, algorithm_type="sde-dpmsolver++"
)

print("Loading Image generation model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = load_sampler(pipe.scheduler.config)

out_folder_path = f"./output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_folder_path, exist_ok=True)


def generate():
    print(f"Generating images from prompt using '{model_id}'")
    for k in range(num_images):
        print(f"Generating image {k}")
        generator = torch.manual_seed(seed + k)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            width=image_size,
            height=image_size,
            num_inference_steps=num_inference_steps,
        ).images[0]
        image_path = f"{out_folder_path}/{k}.jpg"
        image.save(image_path)


generate()
