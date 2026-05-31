from typing import Optional

import torch
import torch.nn.functional as F


def _ssim_fallback(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    mu1 = F.avg_pool2d(img1, window_size, 1, pad)
    mu2 = F.avg_pool2d(img2, window_size, 1, pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, pad) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean()


def masked_rgb_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diff = torch.abs(rendered - target)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[0] == 1 and diff.shape[0] == 3:
            mask = mask.expand_as(diff)
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom
    return diff.mean()


def masked_depth_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diff = torch.abs(rendered - target)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom
    return diff.mean()


def masked_depth_relative_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diff = torch.abs(rendered - target) / target.abs().clamp(min=1e-3)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom
    return diff.mean()


def masked_ssim(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, window_size: int = 7) -> torch.Tensor:
    try:
        from thirdparty.gaussian_splatting.utils.loss_utils import ssim
        value = ssim(rendered, target)
    except Exception:
        value = _ssim_fallback(rendered, target, window_size=window_size)

    if mask is None:
        return 1.0 - value
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return (1.0 - value) * mask.float().mean().clamp(min=1e-6)


def masked_gradient_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if rendered.ndim == 3:
        rendered = rendered.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    def grad_xy(img: torch.Tensor):
        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = img[..., 1:, :] - img[..., :-1, :]
        return gx, gy

    rgx, rgy = grad_xy(rendered)
    tgx, tgy = grad_xy(target)
    diff_x = torch.abs(rgx - tgx)
    diff_y = torch.abs(rgy - tgy)

    if mask is None:
        return 0.5 * (diff_x.mean() + diff_y.mean())

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] == 1 and diff_x.shape[1] != 1:
        mask = mask.expand(-1, diff_x.shape[1], -1, -1)

    mask_x = mask[..., :, 1:]
    mask_y = mask[..., 1:, :]
    loss_x = (diff_x * mask_x).sum() / mask_x.sum().clamp(min=1.0)
    loss_y = (diff_y * mask_y).sum() / mask_y.sum().clamp(min=1.0)
    return 0.5 * (loss_x + loss_y)


def masked_depth_gradient_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if rendered.ndim == 2:
        rendered = rendered.unsqueeze(0).unsqueeze(0)
    elif rendered.ndim == 3:
        rendered = rendered.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)

    def grad_xy(img: torch.Tensor):
        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = img[..., 1:, :] - img[..., :-1, :]
        return gx, gy

    rgx, rgy = grad_xy(rendered)
    tgx, tgy = grad_xy(target)
    diff_x = torch.abs(rgx - tgx)
    diff_y = torch.abs(rgy - tgy)

    if mask is None:
        return 0.5 * (diff_x.mean() + diff_y.mean())

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    mask_x = mask[..., :, 1:]
    mask_y = mask[..., 1:, :]
    loss_x = (diff_x * mask_x).sum() / mask_x.sum().clamp(min=1.0)
    loss_y = (diff_y * mask_y).sum() / mask_y.sum().clamp(min=1.0)
    return 0.5 * (loss_x + loss_y)


def apply_exposure(image: torch.Tensor, exposure_a: torch.Tensor, exposure_b: torch.Tensor) -> torch.Tensor:
    return torch.exp(exposure_a) * image + exposure_b
