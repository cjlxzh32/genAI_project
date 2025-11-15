# Qwen-Image Edit

## Environment Setup

Follow the official [Qwen-Image repository](https://github.com/QwenLM/Qwen-Image) for detailed instructions on setting up the environment.

You need to clone the repository and put the `qwen_image_edit.py` script in the root directory of the cloned repository.

## Experiment Instructions

```bash
python qwen-edit.py \
--input_json <path_to_input_json> \
--input_folder <path_to_input_folder> \
--output_folder <path_to_output_folder> \
--num_inference_steps 50 \
--true_cfg_scale 4.0
```
