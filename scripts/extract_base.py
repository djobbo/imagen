import os
from typing import List, Tuple
import shutil

__dirname = os.path.dirname(__file__)

CONVERT_DIFFUSION_SCRIPT = os.path.join(
    __dirname, "./convert_original_stable_diffusion_to_diffusers.py"
)
CONVERT_LORA_SCRIPT = os.path.join(
    __dirname, "./convert_lora_safetensor_to_diffusers.py"
)

"""
Extracts a model from a base model and a list of LoRA modifiers

base_model_path: Path to the model that will be used to generate the images
loras: Path and alpha of all the LoRA modifiers
output_dir: Path for the output model
output_model_name: Name of the output model
tmp_dir: Path to the temporary directory where the intermediate models will be stored
"""


def extract(
    base_model_path: str,
    loras: List[Tuple[str, float]],
    output_model_name: str,
    output_dir: str,
    tmp_dir: str = "./tmp",
):
    print(f"Generating model '{output_model_name}'")

    os.makedirs(tmp_dir, exist_ok=True)

    # If base model is a safetensors file, extract it
    if base_model_path.endswith(".safetensors"):
        print(f"Extracting base model '{base_model_path}' from .safetensors file")
        extracted_model_path = os.path.join(tmp_dir, f"{output_model_name}_base")
        os.system(
            f'python {CONVERT_DIFFUSION_SCRIPT} --from_safetensors --checkpoint_path="{base_model_path}" --dump_path="{extracted_model_path}" --device="cuda:0"'
        )
        base_model_path = extracted_model_path
    elif not os.path.isdir(base_model_path):
        raise Exception(
            "Invalid base_model_path, must be a .safetensors file or a model folder"
        )

    # Apply all the LoRA modifiers
    for i, (lora_path, alpha) in enumerate(loras):
        print(f"Applying LoRA modifier {lora_path} with alpha={alpha}")
        extracted_model_path = os.path.join(tmp_dir, f"{output_model_name}_lora_{i}")
        alpha_str = str(round(alpha, 2))
        os.system(
            f'python {CONVERT_LORA_SCRIPT} --base_model_path="{base_model_path}" --checkpoint_path="{lora_path}" --dump_path="{extracted_model_path}" --alpha={alpha_str} --device="cuda:0"'
        )
        base_model_path = extracted_model_path

    # rename the final model
    os.rename(base_model_path, output_model_name)

    # Move the final model to the output folder
    os.makedirs(output_dir, exist_ok=True)
    shutil.move(output_model_name, os.path.join(output_dir, output_model_name))
    shutil.rmtree(tmp_dir)

    print(
        f"Done generating model '{output_model_name}' ({output_dir}/{output_model_name})"
    )
