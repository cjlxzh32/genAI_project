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


if __name__ == '__main__':
    guidance_scale=1.5
    num_inference_steps=28
    lpips_model = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    else:
        lpips_model = lpips_model.cpu()

    ir_model = IR.load("ImageReward-v1.0")
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

            curr_output = {
                'task': task,
                'image_id': image_id,
                'lpips': lpips_score,
                'ssim': ssim_score,
                'psnr': psnr_score,
                'image_reward': ir_score,
                'fid': None
            }
            print(f"\t\t* {curr_output}")
            task_results.append(curr_output)

        if task_results:
            avg_lpips = sum(r['lpips'] for r in task_results) / len(task_results)
            avg_ssim = sum(r['ssim'] for r in task_results) / len(task_results)
            avg_psnr = sum(r['psnr'] for r in task_results) / len(task_results)
            avg_ir = sum(r['image_reward'] for r in task_results) / len(task_results)

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
                'fid': fid_score
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'flux_kontext/evaluation_results_{task}_steps_{num_inference_steps}_guidance_{guidance_scale}.csv', index=False)
        print("\n>> Evaluation complete.")
        # print(f"\t> Results saved to `evaluation_results_{task}.csv`.")