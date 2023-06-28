from scripts.extract_base import extract

base_model_path = "models/base_model"

loras = [
    ("tensors/LoRA_1", 0.90),
    ("tensors/LoRA_2", 0.80),
]

output_model_name = "my_custom_model"

extract(
    base_model_path=base_model_path,
    loras=loras,
    output_model_name=output_model_name,
    output_dir="./models",
    tmp_dir="./tmp",
)
