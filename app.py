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

# Put your safetensors's model below
model_id = "models/your_custom_model"

# Put any integer here (same seed = same type of generated images)
seed = 3

# AI's intructions to generate images (How to prompt: https://stable-diffusion-art.com/prompt-guide/)
prompt = ""

# What you wouldn't to be included in images
negative_prompt = ""

# Image resolution
image_size = 768

# Number of setps (higher takes more time but should provide better results)
num_inference_steps = 25

# Number of generated images
num_images = 10

###############################################################
# /!\ Do not modify lines below, unless you know what you do. #
###############################################################

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

max_length = pipe.tokenizer.model_max_length

input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

negative_ids = pipe.tokenizer(
    "",
    truncation=False,
    padding="max_length",
    max_length=input_ids.shape[-1],
    return_tensors="pt",
).input_ids
negative_ids = negative_ids.to("cuda")

concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(pipe.text_encoder(input_ids[:, i : i + max_length])[0])
    neg_embeds.append(pipe.text_encoder(negative_ids[:, i : i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

out_folder_path = f"./output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_folder_path, exist_ok=True)

def generate():
    print(f"Generating images from prompt using '{model_id}'")
    for k in range(num_images):
        print(f"Generating image {k}")
        generator = torch.manual_seed(seed + k)
        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            width=image_size,
            height=image_size,
            num_inference_steps=num_inference_steps,
        ).images[0]
        image_path = f"{out_folder_path}/{k}.jpg"
        image.save(image_path)

generate()
