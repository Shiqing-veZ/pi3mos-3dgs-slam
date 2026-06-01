from math import exp
from typing import Optional

import torch
import torch.nn.functional as F


def _ensure_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(0)
    return image


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = window_size // 2
    gauss = torch.tensor(
        [exp(-((x - center) ** 2) / float(2 * sigma**2)) for x in range(window_size)],
        device=device,
        dtype=dtype,
    )
    gauss = gauss / gauss.sum().clamp(min=1e-8)
    return gauss


def _create_ssim_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window_1d = _gaussian_window(window_size, 1.5, device=device, dtype=dtype).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _prepare_weight(
    weight: Optional[torch.Tensor],
    ref: torch.Tensor,
    expand_channels: bool = True,
) -> Optional[torch.Tensor]:
    if weight is None:
        return None

    ref = _ensure_batched_image(ref)
    if weight.ndim == 2:
        weight = weight.unsqueeze(0).unsqueeze(0)
    elif weight.ndim == 3:
        if weight.shape[0] in (1, ref.shape[1]):
            weight = weight.unsqueeze(0)
        else:
            weight = weight.unsqueeze(1)
    elif weight.ndim != 4:
        raise ValueError(f"Unsupported weight shape {tuple(weight.shape)}")

    weight = weight.to(device=ref.device, dtype=ref.dtype)

    if weight.shape[0] == 1 and ref.shape[0] != 1:
        weight = weight.expand(ref.shape[0], -1, -1, -1)

    if expand_channels and weight.shape[1] == 1 and ref.shape[1] != 1:
        weight = weight.expand(-1, ref.shape[1], -1, -1)

    return weight


def _weighted_mean(value: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    if weight is None:
        return value.mean()
    return (value * weight).sum() / weight.sum().clamp(min=1.0)


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    img1 = _ensure_batched_image(img1)
    img2 = _ensure_batched_image(img2)

    channel = img1.shape[1]
    window = _create_ssim_window(window_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean(dim=1, keepdim=True)


def masked_rgb_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)
    diff = torch.abs(rendered - target)
    weight = _prepare_weight(mask, diff, expand_channels=True)
    return _weighted_mean(diff, weight)


def masked_depth_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)
    diff = torch.abs(rendered - target)
    weight = _prepare_weight(mask, diff, expand_channels=True)
    return _weighted_mean(diff, weight)


def masked_depth_relative_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)
    diff = torch.abs(rendered - target) / target.abs().clamp(min=1e-3)
    weight = _prepare_weight(mask, diff, expand_channels=True)
    return _weighted_mean(diff, weight)


def masked_ssim(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)
    ssim_loss_map = 1.0 - _ssim_map(rendered, target, window_size=window_size)
    weight = _prepare_weight(mask, ssim_loss_map, expand_channels=False)
    if weight is not None and weight.shape[1] != 1:
        weight = weight.mean(dim=1, keepdim=True)
    return _weighted_mean(ssim_loss_map, weight)


def masked_gradient_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)

    def grad_xy(img: torch.Tensor):
        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = img[..., 1:, :] - img[..., :-1, :]
        return gx, gy

    rgx, rgy = grad_xy(rendered)
    tgx, tgy = grad_xy(target)
    diff_x = torch.abs(rgx - tgx)
    diff_y = torch.abs(rgy - tgy)

    weight = _prepare_weight(mask, rendered, expand_channels=True)
    mask_x = None if weight is None else weight[..., :, 1:]
    mask_y = None if weight is None else weight[..., 1:, :]
    loss_x = _weighted_mean(diff_x, mask_x)
    loss_y = _weighted_mean(diff_y, mask_y)
    return 0.5 * (loss_x + loss_y)


def masked_depth_gradient_l1(rendered: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rendered = _ensure_batched_image(rendered)
    target = _ensure_batched_image(target)

    def grad_xy(img: torch.Tensor):
        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = img[..., 1:, :] - img[..., :-1, :]
        return gx, gy

    rgx, rgy = grad_xy(rendered)
    tgx, tgy = grad_xy(target)
    diff_x = torch.abs(rgx - tgx)
    diff_y = torch.abs(rgy - tgy)

    weight = _prepare_weight(mask, rendered, expand_channels=True)
    mask_x = None if weight is None else weight[..., :, 1:]
    mask_y = None if weight is None else weight[..., 1:, :]
    loss_x = _weighted_mean(diff_x, mask_x)
    loss_y = _weighted_mean(diff_y, mask_y)
    return 0.5 * (loss_x + loss_y)


def apply_exposure(image: torch.Tensor, exposure_a: torch.Tensor, exposure_b: torch.Tensor) -> torch.Tensor:
    return torch.exp(exposure_a) * image + exposure_b
