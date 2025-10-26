import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import ImageReward as IR
import torch_fidelity
from tqdm import tqdm
import clip
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


# todo - plz follow the following instructions, it takes a while for installation:
# todo - 1. ENV INITIALIZATION -------------------------------------------------------------
""" 
    % (Run in Terminal with conda)
    cd ./task_scoring
    conda create --name img_bench python=3.10 -y
    conda activate img_bench
    pip install -r requirements.txt
    python benching.py
"""

# todo - 2. FOLDER STRUCTURE ----------------------------------------------------------------
"""
    % (Make sure ur directory follow the expected structure as follows)
        -> task_scoring  
            ├── edited_images              (to be added, should be in the same format as `task_images`)
            │   ├── adding
            │   ├── color_changing
            │   └── replacing
            ├── task_images                (copy from `UltraEdit_inTasks`)
            │   ├── adding  
            │   ├── color_changing  
            │   └── replacing  
            ├── task_prompts               (copy from `UltraEdit_inTasks`)
            │   ├── adding.csv  
            │   ├── color_changing.csv  
            │   └── replacing.csv  
            ├── evaluation_results.csv     (to be generated)
            ├── requirements.txt           (environment setup)
            └── benching.py                (this script)     
"""
# todo - 3. OTHER NOTES ----------------------------------------------------------------
#   It takes a while to compute the FID score, from my experiences, it may take 10-20 mins if you use 1000 pairs of images;
#   U can find all sample scores in the output file (based on task), and the last row in the CSV file indicates the mean scores;
#   FID has no individual score, only the overall score;
# todo --------------------------------------------------------------------------------------

def compute_lpips(img1_path, img2_path, lpips_model):
    # Preprocess transform for LPIPS
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)
    if torch.cuda.is_available():
        t1, t2 = t1.cuda(), t2.cuda()
    with torch.no_grad():
        score = lpips_model(t1, t2).item()
    return score


def compute_ssim(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    return psnr(img1, img2, data_range=255)


def compute_image_reward(img_path, prompt, model):
    # model = get_ir_model()
    img = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        score = model.score(prompt, [img])
        if isinstance(score, (list, tuple)):
            score = score[0]
    return float(score)


def compute_fid(dir1, dir2):
    metrics = torch_fidelity.calculate_metrics(
        input1=dir1,
        input2=dir2,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=False
    )
    return max(metrics['frechet_inception_distance'], 0.0)

def compute_clip_t2i(clip_model, clip_processor, prompt, edit_path):
    with torch.no_grad():
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        # text_tokens = clip.tokenize([text]).to(device)
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text_tokens)
        # # image_features /= image_features.norm(dim=-1, keepdim=True)
        # # text_features /= text_features.norm(dim=-1, keepdim=True)
        # # similarity = (image_features @ text_features.T)
        # similarity = F.cosine_similarity(image_features, text_features, dim=1)
        image = Image.open(edit_path).convert('RGB')
        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        similarity_score = F.cosine_similarity(image_features, text_features, dim=1)

    return similarity_score.item()

def compute_clip_i2i(clip_model, clip_processor, image_path1, image_path2):
    # with torch.no_grad():
    #     image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    #     image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
    #     image_features1 = model.encode_image(image1)
    #     image_features2 = model.encode_image(image2)
    #     similarity = F.cosine_similarity(image_features1, image_features2, dim=1)
    with torch.no_grad():
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')
        inputs = clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
        image_features = clip_model.get_image_features(**inputs)
        feat1 = image_features[0]
        feat2 = image_features[1]
        similarity_score = F.cosine_similarity(feat1, feat2, dim=0)
    return similarity_score.item()

if __name__ == '__main__':
    guidance_scale=2.5
    num_inference_steps=50
    device = torch.device("cuda")
    lpips_model = lpips.LPIPS(net='vgg')
    lpips_model = lpips_model.to(device)

    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    ir_model = IR.load("ImageReward-v1.0", device=device)
    # for task in ['adding', 'color_changing', 'replacing']:
    for task in ['adding']:
        results = []
        print(f">> Benching task {task}:")
        csv_path = os.path.join('task_scoring/task_prompts', f'{task}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"csv file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None, names=['image_id', 'prompt'])
        orig_dir = os.path.join('task_scoring/task_images', task)
        edit_dir = os.path.join(f'flux_kontext/output/steps_{num_inference_steps}_guidance_{guidance_scale}', task)

        task_results = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if idx == 0:
                continue
            image_id = row['image_id']
            prompt = row['prompt']
            print(f"\n\t> Processing image id={image_id}")

            orig_path = os.path.join(orig_dir, f'{image_id}.png')
            edit_path = os.path.join(edit_dir, f'{image_id}.png')
            if not os.path.exists(orig_path) or not os.path.exists(edit_path):
                continue

            lpips_score = compute_lpips(orig_path, edit_path, lpips_model)
            ssim_score = compute_ssim(orig_path, edit_path)
            psnr_score = compute_psnr(orig_path, edit_path)
            ir_score = compute_image_reward(edit_path, prompt, ir_model)
            clip_t2i_score = compute_clip_t2i(clip_model, clip_processor, prompt, edit_path)
            clip_i2i_score = compute_clip_i2i(clip_model, clip_processor, orig_path, edit_path)

            curr_output = {
                'task': task,
                'image_id': image_id,
                'lpips': lpips_score,
                'ssim': ssim_score,
                'psnr': psnr_score,
                'image_reward': ir_score,
                'i2i_clip': clip_i2i_score,
                'i2t_clip': clip_t2i_score,
                'fid': None
            }
            print(f"\t\t* {curr_output}")
            task_results.append(curr_output)

        if task_results:
            avg_lpips = sum(r['lpips'] for r in task_results) / len(task_results)
            avg_ssim = sum(r['ssim'] for r in task_results) / len(task_results)
            avg_psnr = sum(r['psnr'] for r in task_results) / len(task_results)
            avg_ir = sum(r['image_reward'] for r in task_results) / len(task_results)
            avg_i2i_clip = sum(r['i2i_clip'] for r in task_results) / len(task_results)
            avg_i2t_clip = sum(r['i2t_clip'] for r in task_results) / len(task_results)

            print(f"\n>> Computing overall FID score (for task=`{task}`)...")
            fid_score = compute_fid(orig_dir, edit_dir)
            print(f"\t> Done.")

            results.extend(task_results)
            results.append({
                'task': task,
                'image_id': 'average',
                'lpips': avg_lpips,
                'ssim': avg_ssim,
                'psnr': avg_psnr,
                'image_reward': avg_ir,
                'i2i_clip': avg_i2i_clip,
                'i2t_clip': avg_i2t_clip,
                'fid': fid_score
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'flux_kontext/evaluation_results_{task}_steps_{num_inference_steps}_guidance_{guidance_scale}.csv', index=False)
        print("\n>> Evaluation complete.")
        # print(f"\t> Results saved to `evaluation_results_{task}.csv`.")