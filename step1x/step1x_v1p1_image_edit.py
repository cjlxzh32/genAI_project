import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from diffusers import Step1XEditPipeline
from diffusers.utils import load_image

pipeline = Step1XEditPipeline.from_pretrained(
    "stepfun-ai/Step1X-Edit-v1p1-diffusers",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(42)


def main(args):
    df = pd.read_json(args.input_json, orient='records')

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for row in tqdm(df.itertuples(), total=len(df)):
        try:
            input_path = str(input_folder / f"{row.ID}.png")
            image = load_image(input_path).convert("RGB")

            output_path = output_folder / f"{row.ID}.png"
            if output_path.exists():
                print(f"Skipping {row.ID} as output already exists.")
                continue

            prompt = row.edit_prompt
            print(f"Processing ID {row.ID} with prompt: {prompt}")

            inputs = {
                "image": image,
                "prompt": prompt,
                "negative_prompt": " ",
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "generator": generator,
            }

            output = pipeline(**inputs)
            output_image = output.images[0]

            output_image.save(str(output_path))

        except Exception as e:
            print(f"Error processing {row.ID}: {e}")


argparser = argparse.ArgumentParser(description="Step1X Edit Inference Script")
argparser.add_argument(
    "--input_json",
    type=str,
    required=True,
    help="Path to the input JSON file containing image IDs and edit prompts.",
)
argparser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Path to the folder containing input images.",
)
argparser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the folder to save edited images.",
)
argparser.add_argument(
    "--num_inference_steps",
    type=int,
    default=28,
    help="Number of inference steps to use for image generation.",
)
argparser.add_argument(
    "--true_cfg_scale",
    type=float,
    default=6.0,
    help="Guidance scale for the diffusion process.",
)

args = argparser.parse_args()
main(args)
