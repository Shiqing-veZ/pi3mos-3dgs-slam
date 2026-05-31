from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class GaussianAnchor:
    frame_id: int
    point_indices: torch.Tensor
    pose_w2c: torch.Tensor
    uv: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    cam_points: Optional[torch.Tensor] = None


@dataclass
class GaussianState:
    device: torch.device
    anchors: Dict[int, GaussianAnchor] = field(default_factory=dict)
    points_by_frame: Dict[int, torch.Tensor] = field(default_factory=dict)
    colors_by_frame: Dict[int, torch.Tensor] = field(default_factory=dict)
    gaussian_count_by_frame: Dict[int, int] = field(default_factory=dict)

    def register_anchor(
        self,
        frame_id: int,
        point_indices: torch.Tensor,
        pose_w2c: torch.Tensor,
        uv: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        cam_points: Optional[torch.Tensor] = None,
    ) -> None:
        self.anchors[frame_id] = GaussianAnchor(
            frame_id=frame_id,
            point_indices=point_indices.detach().long().cpu(),
            pose_w2c=pose_w2c.detach().cpu(),
            uv=None if uv is None else uv.detach().cpu(),
            depth=None if depth is None else depth.detach().cpu(),
            cam_points=None if cam_points is None else cam_points.detach().cpu(),
        )

    def add_initial_points(self, frame_id: int, points: torch.Tensor, colors: torch.Tensor) -> None:
        if points.numel() == 0:
            return
        self.points_by_frame[frame_id] = points.detach().cpu()
        self.colors_by_frame[frame_id] = colors.detach().cpu()
        self.gaussian_count_by_frame[frame_id] = int(points.shape[0])

    def update_anchor_pose(self, frame_id: int, pose_w2c: torch.Tensor) -> None:
        anchor = self.anchors.get(frame_id)
        if anchor is None:
            return
        anchor.pose_w2c = pose_w2c.detach().cpu()

    @property
    def num_initial_points(self) -> int:
        return int(sum(p.shape[0] for p in self.points_by_frame.values()))

    @property
    def num_frames(self) -> int:
        return int(len(self.points_by_frame))

    def summary(self) -> Dict[str, int]:
        return {
            "num_frames": int(self.num_frames),
            "num_anchors": int(len(self.anchors)),
            "num_initial_points": int(self.num_initial_points),
            "num_gaussian_counts": int(sum(self.gaussian_count_by_frame.values())),
        }
