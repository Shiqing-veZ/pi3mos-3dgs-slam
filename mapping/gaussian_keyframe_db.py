from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .gaussian_frame import GaussianFrame


@dataclass
class GaussianKeyframeDB:
    window_size: int = 10
    static_ratio_thresh: float = 0.6
    confidence_thresh: float = 0.1
    min_frame_gap: int = 1
    min_motion_score: float = 0.0
    min_translation: float = 0.05
    force_max_interval: int = 8
    frames: Dict[int, GaussianFrame] = field(default_factory=dict)
    keyframe_ids: List[int] = field(default_factory=list)
    latest_pose_by_frame: Dict[int, object] = field(default_factory=dict)

    def _relative_translation(self, ref_pose, cand_pose) -> float:
        ref_center = self._camera_center(ref_pose)
        cand_center = self._camera_center(cand_pose)
        if ref_center is None or cand_center is None:
            return 0.0
        return float(np.linalg.norm(ref_center - cand_center))

    def should_accept(self, frame: GaussianFrame) -> bool:
        if not self.keyframe_ids:
            return True

        last_kf = self.keyframe_ids[-1]
        gap = frame.frame_id - last_kf
        if gap < self.min_frame_gap:
            return False

        static_ratio = frame.static_ratio
        if static_ratio is not None and static_ratio < self.static_ratio_thresh:
            return False

        conf = frame.confidence_mean
        if conf is not None and conf < self.confidence_thresh * 0.5:
            return False

        motion_score = float(frame.metadata.get("motion_score", 1.0))
        pose_delta = self._relative_translation(
            self.frames[last_kf].pose_w2c if last_kf in self.frames else None,
            frame.pose_w2c,
        )

        if gap >= self.force_max_interval:
            return True

        if pose_delta >= self.min_translation:
            return True

        if frame.is_keyframe_candidate and motion_score >= self.min_motion_score:
            return True

        return False

    def register(self, frame: GaussianFrame) -> bool:
        self.frames[frame.frame_id] = frame
        if self.should_accept(frame):
            self.keyframe_ids.append(frame.frame_id)
            return True
        return False

    def update_pose(self, frame_id: int, pose_w2c) -> None:
        if frame_id in self.frames:
            self.frames[frame_id].pose_w2c = pose_w2c
        self.latest_pose_by_frame[frame_id] = pose_w2c

    def _camera_center(self, pose_w2c):
        if pose_w2c is None:
            return None
        if isinstance(pose_w2c, np.ndarray):
            if pose_w2c.shape != (4, 4):
                return None
            return np.linalg.inv(pose_w2c)[:3, 3]
        if torch.is_tensor(pose_w2c):
            if pose_w2c.shape[-2:] != (4, 4):
                return None
            return torch.linalg.inv(pose_w2c)[:3, 3].detach().cpu().numpy()
        return None

    def _pose_score(self, ref_pose, cand_pose) -> float:
        ref_center = self._camera_center(ref_pose)
        cand_center = self._camera_center(cand_pose)
        if ref_center is None or cand_center is None:
            return float("inf")
        return float(np.linalg.norm(ref_center - cand_center))

    def select_active_window(self, current_frame_id: Optional[int] = None) -> List[int]:
        if not self.keyframe_ids:
            return []

        ordered = list(self.keyframe_ids)
        if current_frame_id is None or current_frame_id not in self.frames:
            return ordered[-self.window_size :] if self.window_size > 0 else ordered

        ref_pose = self.frames[current_frame_id].pose_w2c
        scored = []
        for fid in ordered:
            cand_pose = self.frames.get(fid).pose_w2c if fid in self.frames else None
            score = self._pose_score(ref_pose, cand_pose)
            scored.append((score, fid))

        scored.sort(key=lambda x: (x[0], x[1]))
        selected = [fid for _, fid in scored[: self.window_size]]
        selected.sort()
        return selected

    def remove_old_frames(self, keep_last_n: Optional[int] = None) -> List[int]:
        keep = self.window_size if keep_last_n is None else keep_last_n
        if keep <= 0 or len(self.keyframe_ids) <= keep:
            return []
        removed = self.keyframe_ids[:-keep]
        self.keyframe_ids = self.keyframe_ids[-keep:]
        return removed

    def get(self, frame_id: int) -> Optional[GaussianFrame]:
        return self.frames.get(frame_id)

    def active_ids(self) -> List[int]:
        return list(self.keyframe_ids)
