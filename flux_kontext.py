import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from optimum.quanto import quantize, freeze, qint8

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
quantize(pipe.transformer, weights=qint8)
freeze(pipe.transformer)
pipe.to("cuda")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]
image.save("result.png")