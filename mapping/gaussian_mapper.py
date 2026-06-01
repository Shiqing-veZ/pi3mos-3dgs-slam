import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_deformer import pose_depth_deform_points, rigid_deform_points
from .gaussian_frame import GaussianFrame
from .gaussian_keyframe_db import GaussianKeyframeDB
from .gaussian_losses import (
    masked_depth_gradient_l1,
    masked_depth_l1,
    masked_depth_relative_l1,
    masked_gradient_l1,
    masked_rgb_l1,
    masked_ssim,
)
from .io_utils import save_ply
from .gaussian_state import GaussianState

class _GaussianViewpoint(nn.Module):
    def __init__(self, frame: GaussianFrame):
        super().__init__()
        from thirdparty.gaussian_splatting.utils.graphics_utils import (
            focal2fov,
            getProjectionMatrix2,
            getWorld2View2,
        )

        self.uid = frame.frame_id
        self.device = frame.image.device
        self.cam_rot_delta = nn.Parameter(torch.zeros(3, device=self.device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, device=self.device))
        self.exposure_a = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.exposure_b = nn.Parameter(torch.tensor([0.0], device=self.device))

        self.grad_mask = None
        self.refresh(frame)

    def refresh(self, frame: GaussianFrame) -> None:
        from thirdparty.gaussian_splatting.utils.graphics_utils import (
            focal2fov,
            getProjectionMatrix2,
        )

        self.original_image = frame.image.float()
        if self.original_image.max() > 1.5:
            self.original_image = self.original_image / 255.0
        self.depth = None if frame.pi3_depth is None else frame.pi3_depth.float()
        self.features = None
        self.pose_w2c = frame.pose_w2c.float().to(self.device)
        self.R = self.pose_w2c[:3, :3]
        self.T = self.pose_w2c[:3, 3]
        self.image_height = int(self.original_image.shape[-2])
        self.image_width = int(self.original_image.shape[-1])
        self.fx = float(frame.intrinsics[0].item())
        self.fy = float(frame.intrinsics[1].item())
        self.cx = float(frame.intrinsics[2].item())
        self.cy = float(frame.intrinsics[3].item())
        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.image_width,
            H=self.image_height,
        ).transpose(0, 1).to(self.device)
        self.update_RT(self.R, self.T)
        self.grad_mask = torch.ones(1, self.image_height, self.image_width, device=self.device)

    def update_RT(self, R, t):
        self.R = R
        self.T = t
        from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2

        self.world_view_transform = getWorld2View2(self.R, self.T).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GaussianMapper:
    def __init__(self, cfg, device: str = "cuda", output_dir: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self.enabled = bool(getattr(cfg, "GAUSSIAN_MAPPING", False))
        self.dynamic_thresh = float(getattr(cfg, "GAUSSIAN_DYNAMIC_THRESH", 0.4))
        self.confidence_thresh = float(getattr(cfg, "GAUSSIAN_INIT_CONF_THRESH", 0.1))
        self.window_size = int(getattr(cfg, "GAUSSIAN_WINDOW_SIZE", 10))
        self.init_window_size = int(getattr(cfg, "GAUSSIAN_INIT_WINDOW_SIZE", max(3, self.window_size // 2)))
        self.min_frame_gap = int(getattr(cfg, "GAUSSIAN_MIN_FRAME_GAP", 1))
        self.deform_on_ba = bool(getattr(cfg, "GAUSSIAN_DEFORM_ON_BA", True))
        self.optimization_iters = int(getattr(cfg, "GAUSSIAN_OPT_ITERS", 1))
        self.init_optimization_iters = int(getattr(cfg, "GAUSSIAN_INIT_OPT_ITERS", max(10, self.optimization_iters * 4)))
        self.final_refine_iters = int(getattr(cfg, "GAUSSIAN_FINAL_REFINEMENT_ITERS", 2000))
        self.densify_every = int(getattr(cfg, "GAUSSIAN_DENSIFY_EVERY", 1500))
        self.opacity_reset_every = int(getattr(cfg, "GAUSSIAN_OPACITY_RESET_EVERY", 750))
        self.prune_thresh = float(getattr(cfg, "GAUSSIAN_PRUNE_THRESH", 0.01))
        self.visibility_reset_threshold = float(getattr(cfg, "GAUSSIAN_VISIBILITY_RESET_THRESHOLD", 0.35))
        self.max_init_points_per_frame = int(getattr(cfg, "GAUSSIAN_MAX_INIT_POINTS_PER_FRAME", 5000))
        self.init_point_sample_ratio = float(getattr(cfg, "GAUSSIAN_INIT_POINT_SAMPLE_RATIO", 1.0))
        self.sh_degree = int(getattr(cfg, "GAUSSIAN_SH_DEGREE", 3))
        self.init_voxel_size = float(getattr(cfg, "GAUSSIAN_INIT_VOXEL_SIZE", 0.015))
        self.new_point_opacity_thresh = float(getattr(cfg, "GAUSSIAN_NEW_POINT_OPACITY_THRESH", 0.35))
        self.new_point_depth_rel_thresh = float(getattr(cfg, "GAUSSIAN_NEW_POINT_DEPTH_REL_THRESH", 0.20))
        self.texture_weight = float(getattr(cfg, "GAUSSIAN_TEXTURE_WEIGHT", 0.75))
        self.new_point_residual_thresh = float(getattr(cfg, "GAUSSIAN_NEW_POINT_RESIDUAL_THRESH", 0.12))
        self.dssim_weight = float(np.clip(getattr(cfg, "GAUSSIAN_DSSIM_WEIGHT", 0.3), 0.0, 1.0))
        self.gradient_loss_weight = float(getattr(cfg, "GAUSSIAN_GRADIENT_LOSS_WEIGHT", 0.15))
        self.max_total_points = int(getattr(cfg, "GAUSSIAN_MAX_TOTAL_POINTS", 30000))
        self.min_points_per_frame = int(getattr(cfg, "GAUSSIAN_MIN_POINTS_PER_FRAME", 400))
        self.depth_loss_weight = float(getattr(cfg, "GAUSSIAN_DEPTH_LOSS_WEIGHT", 0.5))
        self.depth_relative_loss_weight = float(getattr(cfg, "GAUSSIAN_DEPTH_RELATIVE_LOSS_WEIGHT", 0.2))
        self.depth_gradient_loss_weight = float(getattr(cfg, "GAUSSIAN_DEPTH_GRADIENT_LOSS_WEIGHT", 0.1))
        self.soft_static_power = float(getattr(cfg, "GAUSSIAN_SOFT_STATIC_POWER", 2.0))
        self.confidence_power = float(getattr(cfg, "GAUSSIAN_CONFIDENCE_POWER", 1.0))
        self.static_weight_floor = float(getattr(cfg, "GAUSSIAN_STATIC_WEIGHT_FLOOR", 0.05))
        self.pose_depth_min_scale = float(getattr(cfg, "GAUSSIAN_POSE_DEPTH_MIN_SCALE", 0.7))
        self.pose_depth_max_scale = float(getattr(cfg, "GAUSSIAN_POSE_DEPTH_MAX_SCALE", 1.35))

        self.keyframes = GaussianKeyframeDB(
            window_size=self.window_size,
            static_ratio_thresh=float(getattr(cfg, "GAUSSIAN_STATIC_RATIO_THRESH", 0.35)),
            confidence_thresh=self.confidence_thresh,
            min_frame_gap=self.min_frame_gap,
            min_motion_score=float(getattr(cfg, "GAUSSIAN_MIN_MOTION_SCORE", 0.0)),
            min_translation=float(getattr(cfg, "GAUSSIAN_MIN_TRANSLATION", 0.05)),
            force_max_interval=int(getattr(cfg, "GAUSSIAN_FORCE_KEYFRAME_EVERY", 8)),
        )
        self.state = GaussianState(device=self.device)
        self.frames: Dict[int, GaussianFrame] = {}
        self.viewpoints: Dict[int, _GaussianViewpoint] = {}
        self.last_synced_pose: Dict[int, torch.Tensor] = {}
        self.map_step = 0
        self.stage = "bootstrapping"

        self.gaussian_model = None
        self.render = None
        self.pipeline = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        self._bootstrap_backend()

    def _bootstrap_backend(self) -> None:
        try:
            from thirdparty.gaussian_splatting.gaussian_renderer import render
            from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

            self.render = render
            self.gaussian_model = GaussianModel(
                sh_degree=self.sh_degree,
                config={
                    "mapping": {
                        "full_resolution": False,
                        "pcd_downsample": 32,
                        "pcd_downsample_init": 16,
                        "point_size": 0.05,
                        "adaptive_pointsize": True,
                        "sensor_type": "monocular",
                    }
                },
            )
            self.gaussian_model.init_lr(6.0)
            train_args = SimpleNamespace(
                percent_dense=0.01,
                position_lr_init=0.00016,
                position_lr_final=0.0000016,
                position_lr_delay_mult=0.01,
                position_lr_max_steps=30000,
                feature_lr=0.0025,
                opacity_lr=0.05,
                scaling_lr=0.001,
                rotation_lr=0.001,
            )
            self.gaussian_model.training_setup(train_args)
            self.gaussian_model.active_sh_degree = self.gaussian_model.max_sh_degree
            self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        except Exception:
            self.render = None
            self.gaussian_model = None

    def _resize_like(self, tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if tensor is None:
            return None
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        resized = F.interpolate(tensor.float(), size=target_hw, mode="bilinear", align_corners=False)
        return resized.squeeze()

    def _depth_edge_mask(self, depth: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        dx = torch.zeros_like(depth)
        dy = torch.zeros_like(depth)
        dx[:, 1:] = torch.abs(depth[:, 1:] - depth[:, :-1])
        dy[1:, :] = torch.abs(depth[1:, :] - depth[:-1, :])
        return (dx + dy) > threshold

    def _texture_map(self, image: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        image = self._normalize_image(image)
        if image.shape[0] == 3:
            gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            gray = image.squeeze(0)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:] = torch.abs(gray[:, 1:] - gray[:, :-1])
        gy[1:, :] = torch.abs(gray[1:, :] - gray[:-1, :])
        tex = gx + gy
        if tuple(tex.shape) != tuple(target_hw):
            tex = self._resize_like(tex, target_hw)
        return tex / tex.amax().clamp(min=1e-6)

    def _residual_map(self, frame: GaussianFrame, points: torch.Tensor) -> Optional[torch.Tensor]:
        if self.gaussian_model is None or self.render is None or frame.image is None:
            return None
        if self.gaussian_model.get_xyz.shape[0] == 0:
            return None
        render_pkg = self._render_frame(frame.frame_id)
        if render_pkg is None:
            return None
        pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt = self._normalize_image(frame.image).to(pred.device)
        if gt.shape != pred.shape:
            gt = F.interpolate(gt.unsqueeze(0), size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        residual = torch.abs(pred - gt).mean(dim=0)
        if tuple(residual.shape) != tuple(points.shape[:2]):
            residual = self._resize_like(residual, points.shape[:2]).to(points.device)
        return residual

    def _frame_point_budget(self, frame: GaussianFrame, candidate_count: int) -> int:
        static_ratio = frame.static_ratio
        if static_ratio is None:
            static_ratio = 1.0
        ratio = max(0.15, min(1.0, float(static_ratio)))
        budget = int(self.max_init_points_per_frame * ratio)
        budget = max(self.min_points_per_frame, budget)
        return min(candidate_count, budget)

    def _enforce_point_budget(self) -> None:
        if self.gaussian_model is None or self.max_total_points <= 0:
            return
        n_points = int(self.gaussian_model.get_xyz.shape[0])
        overflow = n_points - self.max_total_points
        if overflow <= 0:
            return

        opacity = self.gaussian_model.get_opacity.detach().squeeze(-1)
        kf_ids = self.gaussian_model.unique_kfIDs.to(device=opacity.device, dtype=torch.float32)
        score = opacity + 1e-4 * kf_ids
        prune_idx = torch.argsort(score)[:overflow]
        prune_mask = torch.zeros(n_points, dtype=torch.bool, device=opacity.device)
        prune_mask[prune_idx] = True
        self.gaussian_model.prune_points(prune_mask)

    def _frame_pose_c2w(self, frame: GaussianFrame) -> torch.Tensor:
        if frame.pose_w2c.shape[-2:] != (4, 4):
            raise ValueError("GaussianFrame.pose_w2c must be a 4x4 matrix.")
        return torch.linalg.inv(frame.pose_w2c)

    def _build_viewpoint(self, frame: GaussianFrame) -> _GaussianViewpoint:
        return _GaussianViewpoint(frame)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        if image.ndim == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        return image.clamp(0.0, 1.0)

    def _apply_exposure(self, image: torch.Tensor, viewpoint: _GaussianViewpoint) -> torch.Tensor:
        return (torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b).clamp(0.0, 1.0)

    def _get_or_create_viewpoint(self, frame: GaussianFrame) -> _GaussianViewpoint:
        viewpoint = self.viewpoints.get(frame.frame_id)
        if viewpoint is None:
            viewpoint = self._build_viewpoint(frame)
            self.viewpoints[frame.frame_id] = viewpoint
        else:
            viewpoint.refresh(frame)
        return viewpoint

    def _static_mask(self, frame: GaussianFrame, target_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
        static_mask = frame.static_mask(self.dynamic_thresh, self.confidence_thresh)
        if static_mask is None:
            static_mask = torch.ones(target_hw, device=device)
        else:
            static_mask = static_mask.to(device=device, dtype=torch.float32)
            if tuple(static_mask.shape) != tuple(target_hw):
                static_mask = self._resize_like(static_mask, target_hw).to(device=device)
            static_mask = (static_mask > 0.5).float()
        return static_mask.unsqueeze(0)

    def _soft_static_weight(self, frame: GaussianFrame, target_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
        weight = torch.ones(target_hw, device=device, dtype=torch.float32)

        if frame.dynamic_mask is not None:
            dyn = frame.dynamic_mask
            if dyn.ndim == 3:
                dyn = dyn.squeeze(0)
            dyn = dyn.to(device=device, dtype=torch.float32)
            if tuple(dyn.shape) != tuple(target_hw):
                dyn = self._resize_like(dyn, target_hw).to(device=device)
            static_prob = (1.0 - dyn.clamp(0.0, 1.0)).pow(self.soft_static_power)
            weight = weight * static_prob

        if frame.confidence is not None:
            conf = frame.confidence
            if conf.ndim == 3:
                conf = conf.squeeze(0)
            conf = conf.to(device=device, dtype=torch.float32)
            if tuple(conf.shape) != tuple(target_hw):
                conf = self._resize_like(conf, target_hw).to(device=device)
            conf = conf.clamp(0.0, 1.0).pow(self.confidence_power)
            weight = weight * conf

        valid_rgb = None
        if frame.image is not None:
            rgb = self._normalize_image(frame.image).to(device)
            if tuple(rgb.shape[-2:]) != tuple(target_hw):
                rgb = F.interpolate(rgb.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
            valid_rgb = (rgb.sum(dim=0) > 0.01).float()
            weight = weight * valid_rgb

        if self.confidence_thresh > 0:
            weight = torch.where(weight > self.static_weight_floor, weight, torch.zeros_like(weight))
        else:
            weight = weight.clamp(min=self.static_weight_floor)
        return weight.unsqueeze(0)

    def _depth_mask(self, depth: Optional[torch.Tensor], target_hw: Tuple[int, int], device: torch.device) -> Optional[torch.Tensor]:
        if depth is None:
            return None
        mask = (depth > 0).float()
        if tuple(mask.shape) != tuple(target_hw):
            mask = self._resize_like(mask, target_hw).to(device=device)
        return (mask > 0.5).float().unsqueeze(0)

    def _window_ids(self, frame_ids: Optional[Iterable[int]] = None, cap_window: bool = True) -> List[int]:
        if frame_ids is None:
            frame_ids = self.keyframes.active_ids()
        ids = [fid for fid in frame_ids if fid in self.frames and fid in self.viewpoints]
        if not cap_window or len(ids) <= self.window_size or self.window_size <= 0:
            return ids
        return ids[-self.window_size :]

    def _current_active_ids(self) -> List[int]:
        return self.keyframes.select_active_window()

    def _optimize_once(self, frame_id: int, initialization: bool = False, update_densify: bool = True) -> bool:
        if self.gaussian_model is None or self.render is None:
            return False
        viewpoint = self._make_viewpoint(frame_id)
        if viewpoint is None:
            return False

        render_pkg = self._render_frame(frame_id)
        if render_pkg is None:
            return False

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        opacity = render_pkg["opacity"]
        viewspace = render_pkg["viewspace_points"]
        visibility = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        visible_ratio = float(visibility.float().mean().item()) if visibility is not None and visibility.numel() > 0 else 1.0

        target = viewpoint.original_image.to(image.device)
        if target.shape != image.shape:
            target = F.interpolate(target.unsqueeze(0), size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        rendered = self._apply_exposure(image, viewpoint)

        frame = self.frames[frame_id]
        static_weight = self._soft_static_weight(frame, image.shape[-2:], image.device)
        depth_mask = self._depth_mask(viewpoint.depth, depth.shape[-2:], depth.device) if viewpoint.depth is not None else None
        gt_depth = viewpoint.depth.to(depth.device) if viewpoint.depth is not None else None
        if gt_depth is not None and gt_depth.shape != depth.shape:
            gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), size=depth.shape[-2:], mode="bilinear", align_corners=False).squeeze()
        depth_weight = static_weight if depth_mask is None else static_weight * depth_mask

        rgb_l1 = masked_rgb_l1(rendered, target, static_weight)
        rgb_ssim = masked_ssim(rendered, target, static_weight)
        loss = (1.0 - self.dssim_weight) * rgb_l1 + self.dssim_weight * rgb_ssim
        loss = loss + self.gradient_loss_weight * masked_gradient_l1(rendered, target, static_weight)
        if gt_depth is not None:
            loss = loss + self.depth_loss_weight * masked_depth_l1(depth, gt_depth, depth_weight)
            loss = loss + self.depth_relative_loss_weight * masked_depth_relative_l1(depth, gt_depth, depth_weight)
            loss = loss + self.depth_gradient_loss_weight * masked_depth_gradient_l1(depth, gt_depth, depth_weight)
        if opacity is not None:
            loss = loss + 0.01 * (1.0 - opacity).abs().mean()

        self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if viewspace is not None and visibility is not None:
            self.gaussian_model.max_radii2D[visibility] = torch.max(
                self.gaussian_model.max_radii2D[visibility], radii[visibility]
            )
            self.gaussian_model.add_densification_stats(viewspace, visibility)
        self.gaussian_model.optimizer.step()
        self.gaussian_model.update_learning_rate(self.map_step + 1)
        self.map_step += 1

        if (
            visible_ratio < self.visibility_reset_threshold
            and self.stage != "bootstrapping"
            and visibility is not None
            and visibility.shape[0] == self.gaussian_model.get_xyz.shape[0]
        ):
            self.gaussian_model.reset_opacity_nonvisible([visibility])

        densified_this_step = False
        if update_densify and self.densify_every > 0 and self.map_step % self.densify_every == 0:
            self.gaussian_model.densify_and_prune(
                max_grad=0.0002,
                min_opacity=self.prune_thresh,
                extent=6.0,
                max_screen_size=None,
            )
            densified_this_step = True
        if (
            self.opacity_reset_every > 0
            and self.map_step % self.opacity_reset_every == 0
            and visibility is not None
            and not densified_this_step
            and visibility.shape[0] == self.gaussian_model.get_xyz.shape[0]
        ):
            self.gaussian_model.reset_opacity_nonvisible([visibility])
        return True

    def _make_gaussian_tensors(self, points_world: torch.Tensor, colors_rgb: torch.Tensor):
        try:
            from thirdparty.simple_knn._C import distCUDA2
            from thirdparty.gaussian_splatting.utils.general_utils import inverse_sigmoid
            from thirdparty.gaussian_splatting.utils.sh_utils import RGB2SH

            fused_color = RGB2SH(colors_rgb)
            sh_dim = (self.gaussian_model.max_sh_degree + 1) ** 2 if self.gaussian_model is not None else 1
            features = torch.zeros((fused_color.shape[0], 3, sh_dim), device=self.device)
            features[:, :, 0] = fused_color
            dist2 = torch.clamp_min(distCUDA2(points_world), 1e-7) * 0.0015
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            scales = torch.clamp(scales, min=float(np.log(0.001)), max=float(np.log(0.01)))
            rots = torch.zeros((points_world.shape[0], 4), device=self.device)
            rots[:, 0] = 1.0
            opacities = inverse_sigmoid(0.2 * torch.ones((points_world.shape[0], 1), device=self.device))
            return features, scales, rots, opacities
        except Exception:
            sh_dim = (self.gaussian_model.max_sh_degree + 1) ** 2 if self.gaussian_model is not None else 1
            features = torch.zeros((points_world.shape[0], 3, sh_dim), device=self.device)
            features[:, :, 0] = colors_rgb
            scales = torch.log(0.0015 * torch.ones((points_world.shape[0], 3), device=self.device))
            rots = torch.zeros((points_world.shape[0], 4), device=self.device)
            rots[:, 0] = 1.0
            opacities = torch.full((points_world.shape[0], 1), -1.38629436, device=self.device)
            return features, scales, rots, opacities

    def _novel_mask(self, frame: GaussianFrame, points: torch.Tensor) -> Optional[torch.Tensor]:
        if self.gaussian_model is None or self.render is None:
            return None
        if self.gaussian_model.get_xyz.shape[0] == 0:
            return None

        render_pkg = self._render_frame(frame.frame_id)
        if render_pkg is None:
            return None

        opacity = render_pkg.get("opacity")
        depth = render_pkg.get("depth")
        if opacity is None or depth is None:
            return None

        target_hw = tuple(points.shape[:2])
        opacity = self._resize_like(opacity, target_hw).to(points.device)
        depth = self._resize_like(depth, target_hw).to(points.device)
        point_depth = points[..., 2]
        valid_render_depth = depth > 1e-6
        rel_depth_err = torch.zeros_like(point_depth)
        rel_depth_err[valid_render_depth] = torch.abs(depth[valid_render_depth] - point_depth[valid_render_depth]) / point_depth[valid_render_depth].clamp(min=1e-3)
        novel = (opacity < self.new_point_opacity_thresh) | (~valid_render_depth) | (rel_depth_err > self.new_point_depth_rel_thresh)
        return novel

    def _voxel_downsample(
        self,
        points_world: torch.Tensor,
        colors_rgb: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points_world.shape[0] <= 1 or self.init_voxel_size <= 0:
            return points_world, colors_rgb

        score = scores
        if score is None:
            score = torch.ones(points_world.shape[0], device=points_world.device)

        pts_np = points_world.detach().cpu().numpy()
        cols_np = colors_rgb.detach().cpu().numpy()
        score_np = score.detach().cpu().numpy()
        vox = np.floor(pts_np / self.init_voxel_size).astype(np.int64)
        order = np.argsort(-score_np)
        seen = set()
        keep = []
        for idx in order.tolist():
            key = tuple(vox[idx].tolist())
            if key in seen:
                continue
            seen.add(key)
            keep.append(idx)
        keep = np.asarray(keep, dtype=np.int64)
        points_ds = torch.from_numpy(pts_np[keep]).to(device=points_world.device, dtype=points_world.dtype)
        colors_ds = torch.from_numpy(cols_np[keep]).to(device=colors_rgb.device, dtype=colors_rgb.dtype)
        return points_ds, colors_ds

    def submit_frame(self, frame: GaussianFrame) -> bool:
        if not self.enabled:
            return False

        frame = frame.to(self.device)
        self.frames[frame.frame_id] = frame
        accepted = self.keyframes.register(frame)
        if accepted:
            self._integrate_keyframe(frame)
            self.optimize_window(self.keyframes.select_active_window(frame.frame_id))
        return accepted

    def _integrate_keyframe(self, frame: GaussianFrame) -> None:
        self._get_or_create_viewpoint(frame)
        if frame.pi3_points is None or frame.image is None:
            return

        points = frame.pi3_points
        if points.ndim == 4:
            points = points.squeeze(0)

        dynamic_mask = frame.dynamic_mask
        if dynamic_mask is not None:
            if dynamic_mask.ndim == 3:
                dynamic_mask = dynamic_mask.squeeze(0)
            if dynamic_mask.shape != points.shape[:2]:
                dynamic_mask = self._resize_like(dynamic_mask, points.shape[:2])
        else:
            dynamic_mask = torch.zeros(points.shape[:2], device=points.device)

        confidence = frame.confidence
        if confidence is not None:
            if confidence.ndim == 3:
                confidence = confidence.squeeze(0)
            if confidence.shape != points.shape[:2]:
                confidence = self._resize_like(confidence, points.shape[:2])
        else:
            confidence = torch.ones(points.shape[:2], device=points.device)

        static_mask = (dynamic_mask < self.dynamic_thresh) & (confidence >= self.confidence_thresh)
        edge_mask = self._depth_edge_mask(points[..., 2])
        if edge_mask is not None:
            static_mask = static_mask & (~edge_mask)
        novel_mask = self._novel_mask(frame, points)
        if novel_mask is not None:
            static_mask = static_mask & novel_mask

        residual_mask = self._residual_map(frame, points)
        if residual_mask is not None:
            texture_mask = self._texture_map(frame.image, points.shape[:2]).to(points.device)
            detail_mask = (
                (dynamic_mask < self.dynamic_thresh)
                & (confidence >= self.confidence_thresh * 0.75)
                & (residual_mask > self.new_point_residual_thresh)
                & (texture_mask > 0.08)
            )
            static_mask = static_mask | detail_mask

        texture_mask = self._texture_map(frame.image, points.shape[:2]).to(points.device)

        if static_mask.sum() == 0:
            return

        pose_c2w = self._frame_pose_c2w(frame)
        flat_idx = torch.nonzero(static_mask.reshape(-1), as_tuple=False).squeeze(-1)
        world_points = points.reshape(-1, 3)[static_mask.reshape(-1)]
        colors = self._normalize_image(frame.image)
        if colors.shape[0] == 3:
            colors = colors.permute(1, 2, 0)
        if colors.shape[:2] != points.shape[:2]:
            colors = F.interpolate(
                colors.permute(2, 0, 1).unsqueeze(0),
                size=points.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        world_colors = colors.reshape(-1, 3)[static_mask.reshape(-1)]
        conf_flat = confidence.reshape(-1)[static_mask.reshape(-1)]
        texture_flat = texture_mask.reshape(-1)[static_mask.reshape(-1)]
        point_score = conf_flat * (1.0 + self.texture_weight * texture_flat)
        if residual_mask is not None:
            residual_flat = residual_mask.reshape(-1)[static_mask.reshape(-1)]
            point_score = point_score * (1.0 + 1.5 * residual_flat)

        if world_points.shape[0] == 0:
            return
        base_budget = self._frame_point_budget(frame, world_points.shape[0])
        if self.init_point_sample_ratio < 1.0:
            keep_n = max(1, int(base_budget * self.init_point_sample_ratio))
        else:
            keep_n = base_budget
        keep_n = min(keep_n, world_points.shape[0])
        if keep_n < world_points.shape[0]:
            topk = torch.topk(point_score, k=keep_n, largest=True).indices
            world_points = world_points[topk]
            world_colors = world_colors[topk]
            conf_flat = conf_flat[topk]
            point_score = point_score[topk]

        rot = pose_c2w[:3, :3]
        trans = pose_c2w[:3, 3]
        world_points = (world_points @ rot.T) + trans
        world_points, world_colors = self._voxel_downsample(world_points, world_colors, point_score)

        if world_points.shape[0] == 0:
            return

        features, scales, rots, opacities = self._make_gaussian_tensors(world_points, world_colors)
        self.gaussian_model.extend_from_pcd(
            world_points,
            features,
            scales,
            rots,
            opacities,
            kf_id=frame.frame_id,
        )
        self.state.add_initial_points(frame.frame_id, world_points, world_colors)
        self.state.register_anchor(
            frame.frame_id,
            flat_idx,
            frame.pose_w2c.detach().cpu(),
            uv=self._flat_indices_to_uv(flat_idx, points.shape[:2]),
            depth=points.reshape(-1, 3)[flat_idx][:, 2],
            cam_points=points.reshape(-1, 3)[flat_idx],
        )
        self.last_synced_pose[frame.frame_id] = frame.pose_w2c.detach().cpu()
        self._enforce_point_budget()

    def _flat_indices_to_uv(self, flat_idx: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        height, width = int(hw[0]), int(hw[1])
        v = torch.div(flat_idx, width, rounding_mode="floor")
        u = flat_idx % width
        return torch.stack([u, v], dim=-1)

    def _sample_frame_depth(self, frame: GaussianFrame, anchor) -> Optional[torch.Tensor]:
        depth_map = frame.pi3_depth
        if depth_map is None or anchor.uv is None:
            return None
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(0)
        uv = anchor.uv.to(depth_map.device)
        h, w = depth_map.shape[-2:]
        u = uv[:, 0].long().clamp(0, w - 1)
        v = uv[:, 1].long().clamp(0, h - 1)
        return depth_map[v, u]

    def _make_viewpoint(self, frame_id: int) -> Optional[_GaussianViewpoint]:
        frame = self.frames.get(frame_id)
        if frame is None:
            return None
        return self._get_or_create_viewpoint(frame)

    def _render_frame(self, frame_id: int):
        if self.render is None or self.gaussian_model is None:
            return None
        viewpoint = self._make_viewpoint(frame_id)
        if viewpoint is None:
            return None
        bg = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        return self.render(viewpoint, self.gaussian_model, self.pipeline, bg)

    def render_frame(self, frame_id: int):
        return self._render_frame(frame_id)

    def render_custom_view(
        self,
        pose_w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        image_hw: Tuple[int, int],
        frame_id: int = -1,
    ):
        if self.render is None or self.gaussian_model is None:
            return None

        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        dummy_frame = GaussianFrame(
            frame_id=int(frame_id),
            timestamp=0.0,
            image=torch.zeros(3, image_h, image_w, device=self.device, dtype=torch.float32),
            pose_w2c=pose_w2c.to(self.device).float(),
            intrinsics=intrinsics.to(self.device).float(),
            pi3_points=None,
            pi3_depth=None,
            dynamic_mask=None,
            confidence=None,
            is_keyframe_candidate=False,
            metadata={"ephemeral": True},
        )
        viewpoint = _GaussianViewpoint(dummy_frame)
        bg = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        return self.render(viewpoint, self.gaussian_model, self.pipeline, bg)

    def _build_exposure_optimizer(self, active_ids: List[int]) -> Optional[torch.optim.Optimizer]:
        params = []
        anchor_id = min(active_ids) if active_ids else None
        for frame_id in active_ids:
            if frame_id == anchor_id:
                continue
            viewpoint = self.viewpoints.get(frame_id)
            if viewpoint is None:
                continue
            params.append({"params": [viewpoint.exposure_a], "lr": 0.01})
            params.append({"params": [viewpoint.exposure_b], "lr": 0.01})
        if not params:
            return None
        return torch.optim.Adam(params)

    def optimize_window(self, frame_ids, iters: Optional[int] = None, refine: bool = False, cap_window: bool = True) -> None:
        if self.gaussian_model is None or self.render is None:
            return
        active_ids = self._window_ids(frame_ids, cap_window=cap_window)
        if not active_ids:
            return

        iterations = self.optimization_iters if iters is None else int(iters)
        exposure_optimizer = self._build_exposure_optimizer(active_ids)

        for _ in range(iterations):
            if len(active_ids) == 1:
                frame_id = active_ids[0]
            else:
                weights = torch.linspace(0.25, 1.0, steps=len(active_ids), device="cpu").numpy()
                weights = weights / weights.sum()
                frame_id = int(np.random.choice(active_ids, p=weights))

            if exposure_optimizer is not None:
                exposure_optimizer.zero_grad(set_to_none=True)
            if not self._optimize_once(frame_id, initialization=self.stage == "bootstrapping", update_densify=not refine):
                return

            if exposure_optimizer is not None:
                exposure_optimizer.step()

        if self.stage == "bootstrapping" and len(self.keyframes.active_ids()) >= self.init_window_size:
            self.stage = "online"

    def initialize_map_opt(self, frame_ids: Optional[Iterable[int]] = None, iters: Optional[int] = None) -> None:
        self.stage = "bootstrapping"
        self.optimize_window(frame_ids if frame_ids is not None else self.keyframes.active_ids(), iters=iters or self.init_optimization_iters, refine=False, cap_window=True)

    def map_opt_online(self, frame_ids: Optional[Iterable[int]] = None, iters: Optional[int] = None) -> None:
        if self.stage == "bootstrapping":
            self.stage = "online"
        if frame_ids is None:
            current_id = self.keyframes.keyframe_ids[-1] if self.keyframes.keyframe_ids else None
            frame_ids = self.keyframes.select_active_window(current_id)
        self.optimize_window(frame_ids, iters=iters or self.optimization_iters, refine=False, cap_window=True)

    def final_refine(self, iters: Optional[int] = None) -> None:
        if self.gaussian_model is None or self.render is None:
            return
        self.stage = "refine"
        refine_iters = self.final_refine_iters if iters is None else int(iters)
        self.sync_window(lambda fid: self.keyframes.latest_pose_by_frame.get(fid, None), self.keyframes.active_ids())
        self.optimize_window(self.keyframes.active_ids(), iters=refine_iters, refine=True, cap_window=False)

    def render_all_frames(self, output_dir: Optional[str] = None) -> int:
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if target_dir is None or self.gaussian_model is None or self.render is None:
            return 0
        target_dir.mkdir(parents=True, exist_ok=True)

        rendered = 0
        bg = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        for frame_id in sorted(self.frames.keys()):
            viewpoint = self._make_viewpoint(frame_id)
            if viewpoint is None:
                continue
            render_pkg = self.render(viewpoint, self.gaussian_model, self.pipeline, bg)
            if render_pkg is None:
                continue
            img = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            try:
                import cv2

                cv2.imwrite(str(target_dir / f"frame_{frame_id:05d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            except Exception:
                from imageio.v2 import imwrite

                imwrite(str(target_dir / f"frame_{frame_id:05d}.png"), img_np)
            rendered += 1
        return rendered

    def sync_pose(self, frame_id: int, pose_w2c: torch.Tensor) -> bool:
        if not self.enabled:
            return False
        frame = self.frames.get(frame_id)
        if frame is None:
            return False
        pose_w2c = pose_w2c.detach().cpu()
        prev = self.last_synced_pose.get(frame_id)
        if prev is not None and torch.allclose(prev, pose_w2c, atol=1e-6):
            return False

        anchor = self.state.anchors.get(frame_id)
        if anchor is not None and self.deform_on_ba:
            old_pose = anchor.pose_w2c.to(self.device)
            new_pose = pose_w2c.to(self.device)
            frame_depth = None
            use_depth_update = False
            if self.gaussian_model is not None:
                frame_depth = self._sample_frame_depth(frame, anchor)
                anchor_cam_points = None if anchor.cam_points is None else anchor.cam_points.to(self.device)
                ref_depth = None if anchor.depth is None else anchor.depth.to(self.device)
                use_depth_update = frame_depth is not None and ref_depth is not None and anchor_cam_points is not None
                if use_depth_update:
                    points_src = self.state.points_by_frame.get(frame_id)
                    if points_src is None:
                        self.gaussian_model.apply_anchor_pose_update(frame_id, old_pose, new_pose)
                    else:
                        updated_xyz, scale_ratio, updated_cam_points = pose_depth_deform_points(
                            points_src.to(self.device),
                            old_pose,
                            new_pose,
                            anchor_cam_points=anchor_cam_points,
                            reference_depth=ref_depth,
                            updated_depth=frame_depth.to(self.device),
                            min_depth_scale=self.pose_depth_min_scale,
                            max_depth_scale=self.pose_depth_max_scale,
                        )
                        self.gaussian_model.apply_anchor_pose_depth_update(frame_id, updated_xyz, scale_ratio=scale_ratio)
                        self.state.points_by_frame[frame_id] = updated_xyz.detach().cpu()
                        anchor.depth = frame_depth.detach().cpu()
                        anchor.cam_points = updated_cam_points.detach().cpu()
                else:
                    self.gaussian_model.apply_anchor_pose_update(frame_id, old_pose, new_pose)
            if frame_id in self.state.points_by_frame:
                if not (use_depth_update and frame_depth is not None):
                    points = self.state.points_by_frame[frame_id].to(self.device)
                    updated = rigid_deform_points(points, old_pose, new_pose)
                    self.state.points_by_frame[frame_id] = updated.detach().cpu()
            self.state.update_anchor_pose(frame_id, pose_w2c)

        frame.pose_w2c = pose_w2c
        self.keyframes.update_pose(frame_id, pose_w2c)
        viewpoint = self.viewpoints.get(frame_id)
        if viewpoint is not None:
            pose_dev = pose_w2c.to(self.device)
            viewpoint.update_RT(pose_dev[:3, :3], pose_dev[:3, 3])

        self.last_synced_pose[frame_id] = pose_w2c
        return True

    def sync_window(self, pose_provider, frame_ids) -> int:
        updated = 0
        for frame_id in frame_ids:
            pose = pose_provider(frame_id)
            if pose is None:
                continue
            if self.sync_pose(frame_id, pose):
                updated += 1
        return updated

    def save(self, output_dir: Optional[str] = None) -> None:
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if target_dir is None:
            return
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.gaussian_model is not None and self.gaussian_model.get_xyz.shape[0] > 0:
            self.gaussian_model.save_ply(str(target_dir / "gaussian_map.ply"))
        elif self.state.points_by_frame:
            points = torch.cat(list(self.state.points_by_frame.values()), dim=0).cpu().numpy()
            colors = torch.cat(list(self.state.colors_by_frame.values()), dim=0).cpu().numpy()
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            save_ply(str(target_dir / "gaussian_init_points"), points.astype(np.float32), colors)
        with (target_dir / "gaussian_state.json").open("w") as f:
            json.dump(
                {
                    "stage": self.stage,
                    "map_step": int(self.map_step),
                    "enabled": bool(self.enabled),
                    "keyframes": self.keyframes.active_ids(),
                    "state": self.state.summary(),
                },
                f,
                indent=2,
            )

    def summary(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "stage": self.stage,
            "map_step": int(self.map_step),
            "num_frames": int(len(self.frames)),
            "num_keyframes": int(len(self.keyframes.active_ids())),
            "num_points": int(0 if self.gaussian_model is None else self.gaussian_model.get_xyz.shape[0]),
            "state": self.state.summary(),
        }
