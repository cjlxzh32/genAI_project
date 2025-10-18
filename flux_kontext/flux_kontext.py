import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from diffusers import FluxKontextPipeline
from optimum.quanto import quantize, freeze, qint8
import pandas as pd
from PIL import Image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
quantize(pipe.transformer, weights=qint8)
freeze(pipe.transformer)
pipe.to("cuda")

guidance_scale=2.5
num_inference_steps=40

root = f"task_scoring"
image_root = f"{root}/task_images"
prompt_root = f"{root}/task_prompts"
save_root = f"flux_kontext/output/steps_{num_inference_steps}_guidance_{guidance_scale}"
os.makedirs(save_root, exist_ok=True)

# for task_name in ["adding", "color_changing", "replacing"]:
for task_name in ["adding"]:
    image_dir = f"{image_root}/{task_name}"
    csv_file = f"{prompt_root}/{task_name}.csv"
    
    # read CSV file
    df = pd.read_csv(csv_file)
    prompt_dict = dict(zip(df["ID"].astype(str), df["edit_prompt"]))
    
    # ouput directory
    out_dir = f"{save_root}/{task_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    for img_id, prompt in prompt_dict.items():
        # image may have many extensions
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            img_file = f"{image_dir}/{img_id}{ext}"
            if os.path.exists(img_file):
                found = True
                break
        
        if not found:
            print(f"⚠️ {task_name}: can't find image {img_id}")
            continue
		
		# save path
        save_path = f"{out_dir}/{img_id}{ext}"
        if(os.path.exists(save_path)):
            continue

        input_image = Image.open(img_file)
        image = pipe(
          image=input_image,
          prompt=prompt,
          guidance_scale=guidance_scale,
          num_inference_steps=num_inference_steps,
          generator=torch.Generator("cpu").manual_seed(42)
        ).images[0].resize((512, 512), Image.Resampling.LANCZOS)

        image.save(save_path)
        print(f"✅ {task_name}: {img_file} -> {save_path}, prompt: {prompt}")
