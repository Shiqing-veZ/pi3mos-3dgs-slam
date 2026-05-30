import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation as R

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


def _build_traj_est(traj_est, timestamps):
    return PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def _tum_line_to_matrix(line):
    data = np.fromstring(line, sep=" ")
    if data.shape[0] < 8:
        raise ValueError(f"Invalid TUM pose line: {line}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(data[4:8]).as_matrix()
    T[:3, 3] = data[1:4]
    return T


def _pose7_to_matrix(pose7):
    pose7 = np.asarray(pose7, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(pose7[3:7]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def _estimate_similarity(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 arrays")
    if src.shape[0] < 3:
        raise ValueError("At least 3 points are required for similarity alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    rot = U @ D @ Vt

    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(S * np.diag(D)) / max(var_src, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def _resolve_output_hw(bridge):
    frames = getattr(bridge.mapper, "frames", {})
    if not frames:
        raise RuntimeError("Gaussian mapper has no frames to infer render resolution")
    first_frame = frames[min(frames.keys())]
    image = first_frame.image
    return int(image.shape[-2]), int(image.shape[-1])


def _load_gt_image(image_path, intrinsic, out_hw, device):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(intrinsic["fx"])
    K[1, 1] = float(intrinsic["fy"])
    K[0, 2] = float(intrinsic["ppx"])
    K[1, 2] = float(intrinsic["ppy"])
    coeffs = np.asarray(intrinsic.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    undistorted = cv2.undistort(img_bgr, K, coeffs)

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if undistorted.shape[0] != out_h or undistorted.shape[1] != out_w:
        undistorted = cv2.resize(undistorted, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).float().permute(2, 0, 1).to(device) / 255.0


def _make_intrinsics_tensor(intrinsic, out_hw, device):
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    src_w = float(intrinsic["width"])
    src_h = float(intrinsic["height"])
    scale_w = out_w / src_w
    scale_h = out_h / src_h
    return torch.tensor(
        [
            float(intrinsic["fx"]) * scale_w,
            float(intrinsic["fy"]) * scale_h,
            float(intrinsic["ppx"]) * scale_w,
            float(intrinsic["ppy"]) * scale_h,
        ],
        device=device,
        dtype=torch.float32,
    )


def _associate_tracking_poses(poses, timestamps, gt_file):
    traj_ref = file_interface.read_tum_trajectory_file(str(gt_file))
    traj_est = _build_traj_est(np.asarray(poses, dtype=np.float64), timestamps)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    if traj_ref.num_poses < 3:
        raise RuntimeError("Not enough matched poses to align tracking and GT for NVS evaluation")
    return traj_ref, traj_est


@torch.no_grad()
def evaluate_nvs(bridge, poses, timestamps, tracking_gt_file, nvs_dir, out_dir, save_vis=True):
    nvs_dir = Path(nvs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    with (nvs_dir / "groundtruth.txt").open("r") as f:
        nvs_pose_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    with (nvs_dir / "per_frame_intrinsics.json").open("r") as f:
        intrins_per_frame = json.load(f)

    traj_ref, traj_est = _associate_tracking_poses(poses, timestamps, tracking_gt_file)
    scale, rot_est_to_gt, trans_est_to_gt = _estimate_similarity(
        traj_est.positions_xyz,
        traj_ref.positions_xyz,
    )
    rot_gt_to_est = rot_est_to_gt.T

    out_hw = _resolve_output_hw(bridge)
    device = bridge.mapper.device

    cal_lpips = None
    cal_lpips_mode = None
    if LearnedPerceptualImagePatchSimilarity is not None:
        try:
            cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
            cal_lpips_mode = "torchmetrics"
        except Exception:
            cal_lpips = None
    if cal_lpips is None and lpips_pkg is not None:
        cal_lpips = lpips_pkg.LPIPS(net="alex").to(device)
        cal_lpips_mode = "lpips"

    psnr_list, ssim_list, lpips_list = [], [], []
    per_frame = []

    for i, line in enumerate(nvs_pose_lines):
        intrinsic = intrins_per_frame[str(i)]
        T_wg_c = _tum_line_to_matrix(line)
        R_wg_c = T_wg_c[:3, :3]
        C_wg = T_wg_c[:3, 3]

        R_we_c = rot_gt_to_est @ R_wg_c
        C_we = rot_gt_to_est @ (C_wg - trans_est_to_gt) / scale
        T_we_c = np.eye(4, dtype=np.float64)
        T_we_c[:3, :3] = R_we_c
        T_we_c[:3, 3] = C_we
        pose_w2c = torch.from_numpy(np.linalg.inv(T_we_c)).to(device=device, dtype=torch.float32)

        intrinsics = _make_intrinsics_tensor(intrinsic, out_hw, device)
        render_pkg = bridge.render_custom_view(pose_w2c, intrinsics, out_hw, frame_id=-(i + 1))
        if render_pkg is None:
            continue
        pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt = _load_gt_image(nvs_dir / "rgb" / f"nvs_{i:05d}.png", intrinsic, out_hw, device)

        mask = gt > 0
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)

        psnr_val = psnr(pred[mask].unsqueeze(0), gt[mask].unsqueeze(0)).item()
        ssim_val = ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_val = float("nan")
        if cal_lpips is not None:
            if cal_lpips_mode == "torchmetrics":
                lpips_val = cal_lpips(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            else:
                lpips_val = cal_lpips(pred.unsqueeze(0) * 2.0 - 1.0, gt.unsqueeze(0) * 2.0 - 1.0).mean().item()

        pred_np = (pred.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(str(render_dir / f"nvs_{i:05d}.png"), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))

        if save_vis:
            gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            vis = np.zeros((gt_bgr.shape[0] * 2 + 20, gt_bgr.shape[1], 3), dtype=np.uint8)
            vis[:gt_bgr.shape[0]] = gt_bgr
            vis[gt_bgr.shape[0] + 20:] = pred_bgr
            cv2.imwrite(str(out_dir / "vis" / f"nvs_{i:05d}.png"), vis)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        per_frame.append(
            {
                "frame": f"nvs_{i:05d}",
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "lpips": float(lpips_val),
            }
        )

    result = {
        "protocol": "wildgs_nvs",
        "mean_psnr": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_lpips": float(np.nanmean(lpips_list)) if np.any(~np.isnan(lpips_list)) else float("nan"),
        "num_frames": int(len(per_frame)),
        "lpips_backend": cal_lpips_mode,
        "alignment_scale": float(scale),
    }
    with (out_dir / "final_result.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (out_dir / "per_frame_result.json").open("w") as f:
        json.dump(per_frame, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gaussian mapping on WildGS-style NVS views")
    parser.add_argument("--traj_file", required=True)
    parser.add_argument("--tracking_gt", required=True)
    parser.add_argument("--nvs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    traj = file_interface.read_tum_trajectory_file(args.traj_file)
    poses = []
    for pos, quat in zip(traj.positions_xyz, traj.orientations_quat_wxyz):
        poses.append([pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]])
    raise SystemExit(
        "This module is intended to be called from demo.py with a live Gaussian backend; "
        "standalone CLI rendering from a saved map is not implemented here."
    )
