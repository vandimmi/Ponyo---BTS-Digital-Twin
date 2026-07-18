import os
import argparse
import torch
import lpips
import numpy as np
from PIL import Image
from math import log10
from pytorch_msssim import ssim

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * log10(PIXEL_MAX / torch.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders_dir", type=str, required=True, help="Directory with rendered images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground truth images")
    parser.add_argument("--psnr_max", type=float, default=35.0, help="Max PSNR for normalization")
    args = parser.parse_args()

    # VGG is the standard backbone used in NeRF, Mip-NeRF 360, and 3DGS benchmarks
    # because it correlates better with human perception of structural changes in novel views
    # compared to AlexNet or SqueezeNet.
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    render_files = sorted([f for f in os.listdir(args.renders_dir) if f.endswith(('.png', '.jpg'))])
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    for f in render_files:
        render_path = os.path.join(args.renders_dir, f)
        gt_path = os.path.join(args.gt_dir, f)
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT image not found for {f}, skipping.")
            continue

        render_img = np.array(Image.open(render_path).convert("RGB")) / 255.0
        gt_img = np.array(Image.open(gt_path).convert("RGB")) / 255.0

        # Convert to BCHW PyTorch tensors
        render_tensor = torch.from_numpy(render_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # Calculate PSNR
        psnr_val = calculate_psnr(render_tensor, gt_tensor)
        
        # Calculate SSIM
        ssim_val = ssim(render_tensor, gt_tensor, data_range=1.0, size_average=True).item()
        
        # Calculate LPIPS (requires range [-1, 1])
        render_tensor_lpips = render_tensor * 2.0 - 1.0
        gt_tensor_lpips = gt_tensor * 2.0 - 1.0
        lpips_val = loss_fn_vgg(render_tensor_lpips, gt_tensor_lpips).item()

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_val
        count += 1

    if count == 0:
        print("No matching images found for evaluation.")
        return

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    # Normalize PSNR
    psnr_norm = max(0.0, min(avg_psnr / args.psnr_max, 1.0))
    
    # Calculate Final Score
    score = 0.4 * (1 - avg_lpips) + 0.3 * avg_ssim + 0.3 * psnr_norm

    print("--- Evaluation Results ---")
    print(f"Images Evaluated: {count}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS (VGG): {avg_lpips:.4f}")
    print(f"PSNR Norm (max={args.psnr_max}): {psnr_norm:.4f}")
    print(f"Final Score: {score:.4f}")

if __name__ == "__main__":
    main()
