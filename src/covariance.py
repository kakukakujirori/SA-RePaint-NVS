from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from kornia.filters import (
    box_blur, gaussian_blur2d,
    get_gaussian_kernel1d, get_box_kernel1d,
    get_gaussian_kernel2d, get_box_kernel2d,
    guided_blur)


class BlurOutput:
    def __init__(self, blurred: torch.Tensor, kernel: torch.Tensor):
        self.blurred = blurred
        self.kernel = kernel


def guided_blur_2D(
        guide: Float[torch.Tensor, "... c h w"],
        input: Float[torch.Tensor, "... c h w"],
        kernel_size: int = 5,
        eps: float = 1e-3,
    ):
    guide_flatten = rearrange(guide, "... c h w -> (...) c h w").float()
    input_flatten = rearrange(input, "... c h w -> (...) c h w").float()
    ret_flatten = guided_blur(guide_flatten, input_flatten, kernel_size, eps)
    return ret_flatten.view_as(guide).to(guide.dtype)


def blur_2D(
        tensor: Float[torch.Tensor, "... c h w"],
        k: int,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
    ) -> tuple[Float[torch.Tensor, "... c h w"], Float[torch.Tensor, "1 2k+1 2k+1"]]:
    kernel_size = 2 * k + 1
    sigma = 0.3 * (k - 1) + 0.8  # NOTE: OpenCV does this

    tensor_reshaped = rearrange(tensor, "... c h w -> (...) c h w")
    if kernel_type == "gaussian":
        ret = gaussian_blur2d(tensor_reshaped, kernel_size, sigma=(sigma, sigma))
        kernel = get_gaussian_kernel2d(kernel_size, sigma=(sigma, sigma), device=tensor.device, dtype=tensor.dtype)
    elif kernel_type == "box":
        ret = box_blur(tensor_reshaped, kernel_size)
        kernel = get_box_kernel2d(kernel_size, device=tensor.device, dtype=tensor.dtype)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    return BlurOutput(blurred=ret.reshape_as(tensor), kernel=kernel)


def blur_3D(
        tensor: Float[torch.Tensor, "... f c h w"],
        ks: int,
        kt: int,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
    ) -> tuple[Float[torch.Tensor, "... f c h w"], Float[torch.Tensor, ""]]:
    s_kernel_size = 2 * ks + 1
    t_kernel_size = 2 * kt + 1

    if kernel_type == "gaussian":
        s_sigma = 0.3 * (ks - 1) + 0.8  # NOTE: OpenCV does this
        t_sigma = 0.3 * (kt - 1) + 0.8  # NOTE: OpenCV does this
        temporal_kernel_1d = get_gaussian_kernel1d(t_kernel_size, t_sigma, device=tensor.device, dtype=tensor.dtype)
        spatial_kernel_2d = get_gaussian_kernel2d((s_kernel_size, s_kernel_size), (s_sigma, s_sigma), device=tensor.device, dtype=tensor.dtype)
        spatial_blur_fn = lambda t: gaussian_blur2d(t, (s_kernel_size, s_kernel_size), (s_sigma, s_sigma))
    elif kernel_type == "box":
        temporal_kernel_1d = get_box_kernel1d(t_kernel_size, device=tensor.device, dtype=tensor.dtype)
        spatial_kernel_2d = get_box_kernel2d(s_kernel_size, device=tensor.device, dtype=tensor.dtype)
        spatial_blur_fn = lambda t: box_blur(t, (s_kernel_size, s_kernel_size))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    kernel_3d = temporal_kernel_1d.view(-1, 1, 1) * spatial_kernel_2d.view(1, s_kernel_size, s_kernel_size)

    # Reshape temporal kernel for 1D convolution
    c, h, w = tensor.shape[-3:]
    conv_temporal_kernel = rearrange(temporal_kernel_1d, "() w -> () () w").expand(c, -1, -1)

    # Apply temporal blur (if kt > 0)
    if kt > 0:
        temp_tensor = rearrange(tensor, "... f c h w -> (... h w) c f")
        padded_tensor = F.pad(temp_tensor, (kt, kt), mode='replicate')  # NOTE: 'reflect' is buggy for large tensors
        temp_blurred = F.conv1d(padded_tensor, conv_temporal_kernel, padding='valid', groups=c)
        temp_blurred = rearrange(temp_blurred, "(b h w) c f -> b f c h w", c=c, h=h, w=w)
        temp_blurred = temp_blurred.view_as(tensor)
    else:
        temp_blurred = tensor

    # Apply spatial blur
    if ks > 0:
        spatially_rearranged = rearrange(temp_blurred, "... f c h w -> (... f) c h w")
        spatially_blurred = spatial_blur_fn(spatially_rearranged)
        final_result = spatially_blurred.view_as(tensor)
    else:
        final_result = temp_blurred

    return BlurOutput(blurred=final_result, kernel=kernel_3d)


def safe_division_2D(
        nunom: Float[torch.Tensor, "b c h w"],
        denom: Float[torch.Tensor, "b c h w"],
        k: int = 1,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
        eps: float = 1e-12,
    ) -> Float[torch.Tensor, "b c h w"]:
    """
    Return A/B, assuming that A/B is theoretically a continuous finite function.
    The solution R is derived as R(p) = argmin_r Σ_{q in W(p)} (A(q) - r * B(q))^2
    It has an analytical solution: R(p) = (Σ_{q in W(p)} A(q)B(q)) / (Σ_{q in W(p)} B(q)^2)
    """
    sum_numom_denom = blur_2D(nunom * denom, k, kernel_type).blurred
    sum_denom_squared = blur_2D(denom * denom, k, kernel_type).blurred
    return sum_numom_denom / (sum_denom_squared + eps)


def safe_division_3D(
        nunom: Float[torch.Tensor, "b f c h w"],
        denom: Float[torch.Tensor, "b f c h w"],
        ks: int = 1,
        kt: int = 1,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
        eps: float = 1e-12,
    ) -> Float[torch.Tensor, "b c h w"]:
    """
    Return A/B, assuming that A/B is theoretically a continuous finite function.
    The solution R is derived as R(p) = argmin_r Σ_{q in W(p)} (A(q) - r * B(q))^2
    It has an analytical solution: R(p) = (Σ_{q in W(p)} A(q)B(q)) / (Σ_{q in W(p)} B(q)^2)
    """
    sum_numom_denom = blur_3D(nunom * denom, ks, kt, kernel_type).blurred
    sum_denom_squared = blur_3D(denom * denom, ks, kt, kernel_type).blurred
    return sum_numom_denom / (sum_denom_squared + eps)


def local_covariance_2D(
        X: Float[torch.Tensor, "... c h w"],
        Y: Float[torch.Tensor, "... c h w"],
        k: int,
        channelwise: bool = False,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
    ) -> Float[torch.Tensor, "... c' h w"]:
    """
    If channelwise is True, the output shape is "... c h w"
    If channelwise is False, the output shape is "... c^2 h w"
    """
    if X.shape != Y.shape:
        raise ValueError("Input tensors must have the same shape.")
    if k < 0:
        raise ValueError("Kernel size k must be a non-negative integer.")

    if channelwise:
        X = rearrange(X, "... c h w -> ... c () h w")
        Y = rearrange(Y, "... c h w -> ... c () h w")

    X_centered = X - blur_2D(X, k, kernel_type).blurred
    Y_centered = Y - blur_2D(Y, k, kernel_type).blurred
    outer = X_centered.unsqueeze(-3) * Y_centered.unsqueeze(-4)  # (..., C, C, H, W)
    outer = rearrange(outer, "... c1 c2 h w -> ... (c1 c2) h w")
    outer_blured = blur_2D(outer, k, kernel_type)
    biased_cov, kernel = outer_blured.blurred, outer_blured.kernel
    bessel_correction = 1.0 / (1.0 - torch.sum(kernel**2))
    unbiased_cov = biased_cov * bessel_correction

    if channelwise:
        unbiased_cov = rearrange(unbiased_cov, "... c () h w -> ... c h w")

    return unbiased_cov


def local_covariance_3D(
        X: Float[torch.Tensor, "... f c h w"],
        Y: Float[torch.Tensor, "... f c h w"],
        k_spatial: int,
        k_temporal: int,
        channelwise: bool = False,
        kernel_type: Literal["gaussian", "box"] = "gaussian",
    ) -> Float[torch.Tensor, "... f c^2 h w"]:
    if X.shape != Y.shape:
        raise ValueError("Input tensors must have the same shape.")
    if k_spatial < 0 or k_temporal < 0:
        raise ValueError("Kernel size k_spatial and k_temporal must be a non-negative integer.")

    if channelwise:
        X = rearrange(X, "... f c h w -> ... c f () h w")
        Y = rearrange(Y, "... f c h w -> ... c f () h w")

    X_centered = X - blur_3D(X, k_spatial, k_temporal, kernel_type).blurred
    Y_centered = Y - blur_3D(Y, k_spatial, k_temporal, kernel_type).blurred
    outer = X_centered.unsqueeze(-3) * Y_centered.unsqueeze(-4)  # (..., F, C, C, H, W)
    outer = rearrange(outer, "... f c1 c2 h w -> ... f (c1 c2) h w")
    outer_blured = blur_3D(outer, k_spatial, k_temporal, kernel_type)
    biased_cov, kernel = outer_blured.blurred, outer_blured.kernel
    bessel_correction = 1.0 / (1.0 - torch.sum(kernel**2))
    unbiased_cov = biased_cov * bessel_correction

    if channelwise:
        unbiased_cov = rearrange(unbiased_cov, "... c f () h w -> ... f c h w")

    return unbiased_cov