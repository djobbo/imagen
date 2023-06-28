# AI Image Generation with Diffusers

## Base setup

### Requirements

- Python 3.10+
- Python virtual environments
- NVIDIA GPU with CUDA support (for faster inference)

### Create a python virtual environment
```sh
python3 -m venv .venv
```

### Activate the virtual environment
```sh
source .venv/bin/activate
```

### Install required dependencies
```sh
pip install -r requirements.txt
```

### Disable telemetry
```sh
export DISABLE_TELEMETRY=YES
```

## Install custom models
*Download `.safetensors` files from civitai.com or any other source and put it inside the `tensors/` folder.*

### Extract model from safetensors file
**Run script/extract-safetensors.sh**
```sh
./extract-safetensors.sh 'your_custom_tensors_filename'
```
> Example: if your file is `my_model.safetensors`, run `./extract-safetensors.sh my_model`


The extracted model will be located in `models/your_custom_model_name/` folder.

### Use model

Inside `app.py`, update `models/your_custom_model` with the path to the extracted model.

You can change the parameters of the model in `app.py` as well (prompt, negative prompt...).

### Run the app
```sh
python app.py
```

Output images will be located in `output/` folder.  
Each run will create a new folder with the current timestamp, amd images will be saved in it.

## Resources

- [CivitAI](https://civitai.com/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [Converting original .safetensors file to model](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)
