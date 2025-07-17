import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ─────────────────────────────────────────────────────────────
# 1. PATHS & SETTINGS
# ─────────────────────────────────────────────────────────────
REAL_TEST_FOLDER = r"C:\Datasets\Celeb-A-Split\test"
GEN_IMAGES_FOLDER = r"C:\Datasets\Denoised from models\SDXL_Enhanced_generated_Celeb-A"
IMG_SIZE = (1024, 1024)
N_SAMPLES = 1  # number of generated samples per real image

# ─────────────────────────────────────────────────────────────
# 2. TRANSFORMS & DEVICE
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_tensor = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

resize_np = transforms.Compose([
    transforms.Resize(IMG_SIZE)
])

fid_metric = FrechetInceptionDistance(normalize=True).to(device)

# ─────────────────────────────────────────────────────────────
# 3. MAIN METRIC FUNCTION
# ─────────────────────────────────────────────────────────────
def evaluate_test_split(real_folder, gen_folder):
    psnr_vals, ssim_vals = [], []

    real_files = sorted([
        f for f in os.listdir(real_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"\n##### HAM TEST SET: {len(real_files)} images #####")

    for real_name in tqdm(real_files):
        stem = os.path.splitext(real_name)[0]
        real_path = os.path.join(real_folder, real_name)

        # ✅ Allow any extension for generated images
        sample_paths = glob.glob(os.path.join(gen_folder, f"{stem}_sample_0.*"))

        if len(sample_paths) == 0:
            print(f"[WARN] No samples found for: {stem}")
            continue

        # Load real image
        real_pil = Image.open(real_path).convert("RGB")
        real_np = np.array(real_pil.resize(IMG_SIZE))

        real_tensor = resize_tensor(real_pil).unsqueeze(0).to(device)
        fid_metric.update(real_tensor, real=True)

        for gen_path in sample_paths:
            gen_pil = Image.open(gen_path).convert("RGB")
            gen_np = np.array(gen_pil.resize(IMG_SIZE))

            gen_tensor = resize_tensor(gen_pil).unsqueeze(0).to(device)
            fid_metric.update(gen_tensor, real=False)

            # PSNR & SSIM
            psnr_val = psnr(real_np, gen_np, data_range=255)
            ssim_val = ssim(real_np, gen_np, channel_axis=2, data_range=255)

            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

            print(f"{real_name} ←→ {os.path.basename(gen_path)} | PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}")

    avg_psnr = np.mean(psnr_vals) if psnr_vals else 0.0
    avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0
    fid_score = fid_metric.compute().item() if psnr_vals else 0.0

    print("\n=== FINAL TEST SUMMARY ===")
    print(f"Avg PSNR : {avg_psnr:.2f}")
    print(f"Avg SSIM : {avg_ssim:.4f}")
    print(f"FID      : {fid_score:.3f}")
    return avg_psnr, avg_ssim, fid_score

# ─────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_test_split(REAL_TEST_FOLDER, GEN_IMAGES_FOLDER)
