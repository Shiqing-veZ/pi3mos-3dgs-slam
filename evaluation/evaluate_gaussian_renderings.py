import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None

try:
    import lpips as lpips_pkg
except Exception:
    lpips_pkg = None


def _load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


@torch.no_grad()
def evaluate(render_dir, gt_dir, out_dir, save_vis=True):
    render_dir = Path(render_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    render_files = {p.stem: p for p in render_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    gt_files = {p.stem: p for p in gt_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    common_keys = sorted(render_files.keys() & gt_files.keys())
    if not render_files:
        raise FileNotFoundError(f"No render images found in {render_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No GT images found in {gt_dir}")
    if not common_keys:
        render_list = sorted(render_files.values())
        gt_list = sorted(gt_files.values())
        n = min(len(render_list), len(gt_list))
        render_pairs = render_list[:n]
        gt_pairs = gt_list[:n]
    else:
        render_pairs = [render_files[k] for k in common_keys]
        gt_pairs = [gt_files[k] for k in common_keys]

    cal_lpips = None
    cal_lpips_mode = None
    if LearnedPerceptualImagePatchSimilarity is not None:
        try:
            cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
            cal_lpips_mode = "torchmetrics"
        except Exception:
            cal_lpips = None
    if cal_lpips is None and lpips_pkg is not None:
        cal_lpips = lpips_pkg.LPIPS(net="alex").to("cuda")
        cal_lpips_mode = "lpips"

    psnr_list, ssim_list, lpips_list = [], [], []
    per_frame = []
    for idx, (pred_path, gt_path) in enumerate(zip(render_pairs, gt_pairs)):
        frame_name = common_keys[idx] if common_keys else pred_path.stem
        pred = _load_rgb(pred_path).cuda()
        gt = _load_rgb(gt_path).cuda()

        if pred.shape != gt.shape:
            gt = torch.nn.functional.interpolate(gt.unsqueeze(0), size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

        mask = (gt > 0).float()
        if mask.sum() == 0:
            mask = torch.ones_like(mask)

        psnr_val = psnr(pred[mask.bool()].unsqueeze(0), gt[mask.bool()].unsqueeze(0)).item()
        ssim_val = ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_val = float("nan")
        if cal_lpips is not None:
            if cal_lpips_mode == "torchmetrics":
                lpips_val = cal_lpips(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            else:
                lpips_val = cal_lpips(pred.unsqueeze(0) * 2.0 - 1.0, gt.unsqueeze(0) * 2.0 - 1.0).mean().item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        per_frame.append(
            {
                "frame": frame_name,
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "lpips": float(lpips_val),
            }
        )

        if save_vis:
            pred_bgr = cv2.cvtColor((pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor((gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            vis = np.zeros((gt_bgr.shape[0] * 2 + 20, gt_bgr.shape[1], 3), dtype=np.uint8)
            vis[:gt_bgr.shape[0]] = gt_bgr
            vis[gt_bgr.shape[0] + 20:] = pred_bgr
            cv2.imwrite(str(out_dir / "vis" / f"{frame_name}.png"), vis)

    result = {
        "mean_psnr": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_lpips": float(np.nanmean(lpips_list)) if np.any(~np.isnan(lpips_list)) else float("nan"),
        "num_frames": int(len(render_pairs)),
        "lpips_backend": cal_lpips_mode,
    }
    with open(out_dir / "final_result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "per_frame_result.json", "w") as f:
        json.dump(per_frame, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rendered images with PSNR/SSIM/LPIPS")
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out_dir", default="eval_results/gaussian_renderings")
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    evaluate(args.render_dir, args.gt_dir, args.out_dir, save_vis=not args.no_vis)
