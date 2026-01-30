"""Taken from https://github.com/princeton-computational-imaging/NSF/blob/main/utils/utils.py
"""
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from kornia.geometry.transform.imgwarp import homography_warp


# spline interpolation utils
def interpolate(signal: Float[torch.Tensor, "batch channels num_control_points"], times: Float[torch.Tensor, "batch num_sampling"]):
    if signal.shape[-1] == 1:
        return signal.squeeze(-1)
    elif signal.shape[-1] == 2:
        return interpolate_linear(signal, times)
    else:
        return interpolate_cubic_hermite(signal, times)


@torch.jit.script
def interpolate_cubic_hermite(signal, times):
    # Interpolate a signal using cubic Hermite splines
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    N = signal.shape[-1]

    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, 0, N - 2)
    left_indices = torch.clamp(indices - 1, 0, N - 1)
    right_indices = torch.clamp(indices + 1, 0, N - 1)
    right_right_indices = torch.clamp(indices + 2, 0, N - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation


@torch.jit.script
def interpolate_linear(signal, times):
    # Interpolate a signal using linear interpolation
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    # Scale times to be between 0 and N - 1
    times_scaled = times * (signal.shape[-1] - 1)

    indices = torch.floor(times_scaled).long()
    right_indices = (indices + 1).clamp(max=signal.shape[-1] - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, indices)
    p1 = torch.gather(signal, -1, right_indices)

    # Linear basis functions
    h00 = (1 - t)
    h01 = t

    interpolation = h00 * p0 + h01 * p1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation


def compute_corner_motion_smoothness(
    homography_matrices: Float[torch.Tensor, "batch num_frames 3 3"],
    order: int = 1,
) -> Float[torch.Tensor, ""]:
    """
    Compute smoothness regularization based on corner point motion:
    1. Transforms 4 image corners through each homography matrix
    2. Computes first/second-order finite differences of corner positions
    3. Returns the mean absolute acceleration as a loss term

    Args:
        homography_matrices: Homography matrices of shape (batch, num_frames, 3, 3)
        order: Order of the finite difference (1 for first-order, 2 for second-order)

    Returns:
        Scalar loss representing mean corner acceleration magnitude
    """
    batch, num_frames, _, _ = homography_matrices.shape

    if num_frames < 3:  # motion smoothness undefined
        return torch.tensor(0.0, device=homography_matrices.device, dtype=homography_matrices.dtype)

    # Define 4 corners in normalized coordinates [-1, 1] x [-1, 1]
    # Shape: (4, 3) in homogeneous coordinates
    corners = torch.tensor([
        [-1.0, -1.0, 1.0],  # top-left
        [ 1.0, -1.0, 1.0],  # top-right
        [ 1.0,  1.0, 1.0],  # bottom-right
        [-1.0,  1.0, 1.0],  # bottom-left
    ], device=homography_matrices.device, dtype=homography_matrices.dtype)  # (4, 3)

    # Apply homography to corners: (H @ p)^T = p^T @ H^T
    # corners (1, 1, 4, 3) broadcasts with H.mT (batch, num_frames, 3, 3)
    transformed_corners_homo = torch.matmul(corners[None, None], homography_matrices.mT)  # (batch, num_frames, 4, 3)

    # Convert from homogeneous to Cartesian coordinates
    transformed_corners = transformed_corners_homo[:, :, :, :2] / (transformed_corners_homo[:, :, :, 2:3] + 1e-8)  # (batch, num_frames, 4, 2)

    # Compute [1/2]-order finite difference
    if order == 1:
        diff = transformed_corners[:, 1:] - transformed_corners[:, :-1]
    elif order == 2:
        diff = transformed_corners[:, 2:, :, :] - 2 * transformed_corners[:, 1:-1, :, :] + transformed_corners[:, :-2, :, :]
    else:
        raise NotImplementedError(f"{order=} unsupported.")

    return diff.abs().mean()


@torch.enable_grad()
def homography_estimation(
        frames_src: Float[torch.Tensor, "batch num_frames c h w"],
        frames_dst: Float[torch.Tensor, "batch num_frames c h w"],
        frames_dst_mask: Float[torch.Tensor, "batch num_frames 1 h w"],
        process_size: int = 128,
        lr: float = 1e-2,
        max_iters: int = 100,
        num_control_points: Optional[int] = None,
        fix_first_frame: bool = True,
        smoothness_weight: float = 0.5,  # regularization to prevent erratic warp
        smoothness_order: int = 2,
        padding_mode: str = "border",  # 'zeros', 'border', 'reflection'
        padding_noise_std: float = 0.0,
        init_homography: Optional[Float[torch.Tensor, "batch num_control_points 3 3"]] = None,
    ):
    batch, num_frames, channel, height, width = frames_src.shape
    assert frames_src.shape == frames_dst.shape, f"{frames_src.shape=}"
    assert frames_dst_mask.shape == (batch, num_frames, 1, height, width) or frames_dst_mask.shape == (batch, num_frames, channel, height, width), f"{frames_dst_mask.shape=}"

    # flatten -> resize -> unflatten
    shrink_scale = process_size / max(height, width)
    frames_src_sh = F.interpolate(frames_src.flatten(0,1), scale_factor=shrink_scale, mode="bilinear")
    frames_dst_sh = F.interpolate(frames_dst.flatten(0,1), scale_factor=shrink_scale, mode="bilinear")
    frames_dst_mask_sh = F.interpolate(frames_dst_mask.flatten(0,1).float(), scale_factor=shrink_scale, mode="area")
    frames_src_sh = rearrange(frames_src_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)
    frames_dst_sh = rearrange(frames_dst_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)
    frames_dst_mask_sh = rearrange(frames_dst_mask_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

    # binarize and expand mask
    frames_dst_mask_sh = frames_dst_mask_sh.expand_as(frames_dst_sh) > 0.5
    height_sh, width_sh = frames_src_sh.shape[-2:]

    # spatially align
    if num_control_points is None:
        num_control_points = num_frames
    else:
        assert num_control_points >= 2
    homography_params = torch.nn.Parameter(torch.zeros(batch, 8, num_control_points).to(frames_src))
    homography_params.data[:, 0, :] = 1.0
    homography_params.data[:, 4, :] = 1.0

    if init_homography is not None:
        init_homography_params = rearrange(init_homography, "b num_control_points m n -> b (m n) num_control_points", m=3, n=3)
        init_homography_params = init_homography_params[:, :8, :] / init_homography_params[:, 8:9, :]  # NOTE: Assume the right bottom is non-zero
        assert init_homography_params.shape == (batch, 8, num_control_points), f"{init_homography_params.shape=}"
        homography_params.data.copy_(init_homography)

    homography_query_times = torch.linspace(0, 1, num_frames).reshape(1, num_frames).expand(batch, -1).to(frames_src)
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([homography_params], lr=lr)

    for iter in range(max_iters):
        optimizer.zero_grad()

        # warp
        homography_all = interpolate(homography_params, homography_query_times)
        assert homography_all.shape == (batch, 8, num_frames)
        M = torch.cat([homography_all, torch.ones_like(homography_all[:, 0:1, :])], dim=1).reshape(batch, 3, 3, num_frames)
        M = M.permute(0, 3, 1, 2).reshape(batch * num_frames, 3, 3)

        src_warped = homography_warp(
            frames_src_sh.reshape(batch * num_frames, channel, height_sh, width_sh),
            M,
            dsize=(height_sh, width_sh),
        ).reshape_as(frames_src_sh)

        # update
        loss_reconst = loss_func(src_warped[frames_dst_mask_sh], frames_dst_sh[frames_dst_mask_sh])
        loss_regularize = compute_corner_motion_smoothness(M.reshape(batch, num_frames, 3, 3), order=smoothness_order)
        loss = loss_reconst + loss_regularize * smoothness_weight
        loss.backward()
        if fix_first_frame:
            homography_params.grad[:, :, 0] = 0.0  # Zero grad for 0-th control point
        optimizer.step()

        # if iter % 100 == 0 or iter == max_iters - 1:
        #     print(f"[homography_estimation] {iter=}, loss_reconst={loss_reconst.item()}, loss_regularize={loss_regularize.item()}")

    with torch.no_grad():
        # apply the optimized warp
        src_warped = homography_warp(
            frames_src.flatten(0, 1),
            M,
            dsize=(height, width),
            padding_mode=padding_mode,
        ).reshape_as(frames_src)

        if padding_noise_std > 0:
            valid_region = homography_warp(
                torch.ones(batch * num_frames, 1, height, width, dtype=frames_src.dtype, device=frames_src.device),
                M,
                dsize=(height, width),
                padding_mode='zeros',
            ).reshape(batch, num_frames, 1, height, width)
            noise = torch.randn_like(src_warped) * padding_noise_std
            src_warped += noise * (1 - valid_region)

    return M, src_warped
