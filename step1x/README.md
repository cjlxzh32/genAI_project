# Bagel

## Environment Setup

Follow the official [Step1X repository](https://github.com/stepfun-ai/Step1X-Edit) for detailed instructions on setting up the environment.

You need to clone the repository and put the `step1x_v1p1_image_edit.py` script in the root directory of the cloned repository.

## Experiment Instructions

```bash
python step1x_v1p1_image_edit.py \
--input_json <path_to_input_json> \
--input_folder <path_to_input_folder> \
--output_folder <path_to_output_folder> \
--num_inference_steps 28 \
--true_cfg_scale 6.0
```
