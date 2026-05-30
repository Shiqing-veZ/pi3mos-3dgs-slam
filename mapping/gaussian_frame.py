from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class GaussianFrame:
    frame_id: int
    timestamp: float
    image: torch.Tensor
    pose_w2c: torch.Tensor
    intrinsics: torch.Tensor
    pi3_points: Optional[torch.Tensor] = None
    pi3_depth: Optional[torch.Tensor] = None
    dynamic_mask: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    is_keyframe_candidate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "GaussianFrame":
        return GaussianFrame(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            image=self.image.to(device),
            pose_w2c=self.pose_w2c.to(device),
            intrinsics=self.intrinsics.to(device),
            pi3_points=None if self.pi3_points is None else self.pi3_points.to(device),
            pi3_depth=None if self.pi3_depth is None else self.pi3_depth.to(device),
            dynamic_mask=None if self.dynamic_mask is None else self.dynamic_mask.to(device),
            confidence=None if self.confidence is None else self.confidence.to(device),
            is_keyframe_candidate=self.is_keyframe_candidate,
            metadata=dict(self.metadata),
        )

    @property
    def static_ratio(self) -> Optional[float]:
        if self.dynamic_mask is None:
            return None
        mask = self.dynamic_mask
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        return float((mask < 0.5).float().mean().item())

    @property
    def confidence_mean(self) -> Optional[float]:
        if self.confidence is None:
            return None
        return float(self.confidence.float().mean().item())

    def static_mask(self, dynamic_thresh: float, confidence_thresh: float = 0.0) -> Optional[torch.Tensor]:
        if self.dynamic_mask is None:
            return None
        mask = self.dynamic_mask
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        static = mask < dynamic_thresh
        if self.confidence is not None:
            conf = self.confidence
            if conf.ndim == 3:
                conf = conf.squeeze(0)
            static = static & (conf >= confidence_thresh)
        return static
