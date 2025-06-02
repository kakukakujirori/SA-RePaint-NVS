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
        acceleration_penalty_weight: float = 0.1,  # regularization to prevent erratic warp
    ):
    batch, num_frames, channel, height, width = frames_src.shape
    assert frames_src.shape == frames_dst.shape, f"{frames_src.shape=}"
    assert frames_dst_mask.shape == (batch, num_frames, 1, height, width), f"{frames_dst_mask.shape=}"

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
        if num_control_points >= 3:
            loss_regularize = (homography_params[:, :, 2:] - 2 * homography_params[:, :, 1:-1] + homography_params[:, :, :-2]).abs().mean() * acceleration_penalty_weight
        else:
            loss_regularize = 0
        loss = loss_reconst + loss_regularize
        loss.backward()
        if fix_first_frame:
            homography_params.grad[:, :, 0] = 0.0  # Zero grad for 0-th control point
        optimizer.step()

        if iter % 100 == 0 or iter == max_iters - 1:
            print(f"[homography_estimation] {iter=}, loss_reconst={loss_reconst.item()}, loss_regularize={loss_regularize.item()}")

    with torch.no_grad():
        # apply the optimized warp
        src_warped = homography_warp(
            frames_src.flatten(0, 1),
            M,
            dsize=(height, width),
            padding_mode="border",
        ).reshape_as(frames_src)

    return M, src_warped
