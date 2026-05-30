from typing import Union

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
