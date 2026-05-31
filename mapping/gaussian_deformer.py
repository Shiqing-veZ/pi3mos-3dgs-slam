from typing import Optional, Union

import numpy as np
import torch


def _as_pose_tensor(pose: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(pose, np.ndarray):
        return torch.from_numpy(pose).float()
    return pose.float()


def rigid_deform_points(
    points: torch.Tensor,
    old_pose_w2c: Union[torch.Tensor, np.ndarray],
    new_pose_w2c: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    if points.numel() == 0:
        return points

    old_pose_w2c = _as_pose_tensor(old_pose_w2c).to(points.device)
    new_pose_w2c = _as_pose_tensor(new_pose_w2c).to(points.device)
    transform = torch.linalg.inv(new_pose_w2c) @ old_pose_w2c

    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    return (transform @ points_h.t()).t()[:, :3]


def pose_depth_deform_points(
    points: torch.Tensor,
    old_pose_w2c: Union[torch.Tensor, np.ndarray],
    new_pose_w2c: Union[torch.Tensor, np.ndarray],
    anchor_cam_points: Optional[torch.Tensor] = None,
    reference_depth: Optional[torch.Tensor] = None,
    updated_depth: Optional[torch.Tensor] = None,
    min_depth_scale: float = 0.7,
    max_depth_scale: float = 1.35,
):
    if points.numel() == 0:
        return points, None

    old_pose_w2c = _as_pose_tensor(old_pose_w2c).to(points.device)
    new_pose_w2c = _as_pose_tensor(new_pose_w2c).to(points.device)
    old_pose_c2w = torch.linalg.inv(old_pose_w2c)
    new_pose_c2w = torch.linalg.inv(new_pose_w2c)

    if anchor_cam_points is None:
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=-1)
        anchor_cam_points = (old_pose_w2c @ points_h.t()).t()[:, :3]
    else:
        anchor_cam_points = anchor_cam_points.to(points.device, dtype=points.dtype)

    scale_ratio = None
    updated_cam_points = anchor_cam_points
    if reference_depth is not None and updated_depth is not None:
        reference_depth = reference_depth.to(points.device, dtype=points.dtype).clamp(min=1e-3)
        updated_depth = updated_depth.to(points.device, dtype=points.dtype).clamp(min=1e-3)
        scale_ratio = (updated_depth / reference_depth).clamp(min=min_depth_scale, max=max_depth_scale)
        depth_scaled = anchor_cam_points.clone()
        depth_scaled[:, :2] = depth_scaled[:, :2] * scale_ratio.unsqueeze(-1)
        depth_scaled[:, 2] = updated_depth
        anchor_cam_points = depth_scaled
        updated_cam_points = depth_scaled

    ones = torch.ones(anchor_cam_points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([anchor_cam_points, ones], dim=-1)
    updated = (new_pose_c2w @ points_h.t()).t()[:, :3]
    return updated, scale_ratio, updated_cam_points
