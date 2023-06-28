# get first two arguments and store them in variables
CHECKPOINT_PATH="tensors/$1.safetensors"
DUMP_PATH="models/$1"

echo "Extracting safetensors from $CHECKPOINT_PATH"

python scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors --checkpoint_path="$CHECKPOINT_PATH" --dump_path="$DUMP_PATH" --device='cuda:0'

echo "Done extracting safetensors from $CHECKPOINT_PATH, saved to $DUMP_PATH"
