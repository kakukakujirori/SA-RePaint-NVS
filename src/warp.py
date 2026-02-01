from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from kornia.geometry.transform import get_perspective_transform
from kornia.geometry.transform.imgwarp import homography_warp


@torch.enable_grad()
def homography_estimation(
        frames_src: Float[torch.Tensor, "batch num_frames c h w"],
        frames_dst: Float[torch.Tensor, "batch num_frames c h w"],
        frames_dst_mask: Float[torch.Tensor, "batch num_frames 1 h w"],
        process_size: int | None = None,
        corner_max_shift_ratio: float = 0.5,  # half of the image size
        loss_type: Literal["cos_sim", "l1", "l2"] = "cos_sim",
        lr: float = 1e-2,
        max_iters: int = 100,
        fix_first_frame: bool = True,
        smoothness_weight: float = 0.5,  # regularization to prevent erratic warp
        smoothness_order: int = 2,
        padding_mode: str = "border",  # 'zeros', 'border', 'reflection'
        padding_noise_std: float = 0.0,
        init_homography: Float[torch.Tensor, "batch num_frames 3 3"] | None = None,
    ):
    batch, num_frames, channel, height, width = frames_src.shape
    assert frames_src.shape == frames_dst.shape, f"{frames_src.shape=}"
    assert frames_dst_mask.shape == (batch, num_frames, 1, height, width) or frames_dst_mask.shape == (batch, num_frames, channel, height, width), f"{frames_dst_mask.shape=}"

    # flatten -> resize -> unflatten
    shrink_scale = process_size / max(height, width) if process_size is not None else 1
    frames_src_sh = F.interpolate(frames_src.flatten(0,1), scale_factor=shrink_scale, mode="bilinear")
    frames_dst_sh = F.interpolate(frames_dst.flatten(0,1), scale_factor=shrink_scale, mode="bilinear")
    frames_dst_mask_sh = F.interpolate(frames_dst_mask.flatten(0,1).float(), scale_factor=shrink_scale, mode="area")
    frames_src_sh = rearrange(frames_src_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)
    frames_dst_sh = rearrange(frames_dst_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)
    frames_dst_mask_sh = rearrange(frames_dst_mask_sh, "(b f) c h w -> b f c h w", b=batch, f=num_frames)

    # binarize and expand mask
    frames_dst_mask_sh = frames_dst_mask_sh.expand_as(frames_dst_sh) > 0.5
    height_sh, width_sh = frames_src_sh.shape[-2:]

    if frames_dst_mask_sh.shape[2] > 1:
        assert torch.allclose(frames_dst_mask_sh, frames_dst_mask_sh[:, :, 0:1])
        frames_dst_mask_sh = frames_dst_mask_sh[:, :, 0]

    # loss and optimizer
    if loss_type.lower() == "l1":
        loss_func = lambda x, y: F.l1_loss(x, y, reduction='none').mean(dim=2)
    elif loss_type.lower() == "l2":
        loss_func = lambda x, y: F.mse_loss(x, y, reduction='none').mean(dim=2)
    elif loss_type.lower() == "cos_sim":
        loss_func = lambda x, y: 1 - F.cosine_similarity(x, y, dim=2)
    else:
        raise NotImplementedError(f"{loss_type=} unsupported.")

    delta = torch.nn.Parameter(torch.zeros(batch, num_frames, 4, 2, device=frames_src.device, dtype=frames_src.dtype))
    optimizer = torch.optim.Adam([delta], lr=lr)

    # base corners
    corners_src = torch.tensor([
        [-1.0, -1.0],  # top-left
        [ 1.0, -1.0],  # top-right
        [ 1.0,  1.0],  # bottom-right
        [-1.0,  1.0],  # bottom-left
    ], device=frames_src.device, dtype=frames_src.dtype).reshape(1, 1, 4, 2).expand(batch, num_frames, 4, 2)
    if init_homography is not None:
        corners_src_hom = torch.cat([corners_src, torch.ones_like(corners_src[:, :, :, 0:1])], dim=-1)
        corners_tgt_hom = torch.matmul(corners_src_hom, init_homography.mT)      # (batch, num_frames, 4, 3)
        corners_tgt = corners_tgt_hom[:, :, :, 0:2] / corners_tgt_hom[:, :, :, 2:3]  # (batch, num_frames, 4, 2)
        delta_constrained = corners_tgt - corners_src
        delta_init = torch.atanh(delta_constrained / (corner_max_shift_ratio * 2))
        if not torch.isfinite(delta_init).all():
            print("[homography_estimation] homography init failed. Increase corner_max_shift_ratio or remove init_homography. Fallback to zero init.")
            delta_init = torch.zeros_like(delta_init)
        delta.data = delta_init

    for iter in range(max_iters):
        optimizer.zero_grad()

        # move corners
        delta_constrained = torch.tanh(delta) * corner_max_shift_ratio * 2
        corners_tgt = corners_src + delta_constrained

        # deduce homography
        M = get_perspective_transform(corners_src.reshape(-1, 4, 2), corners_tgt.reshape(-1, 4, 2))

        # warp
        src_warped = homography_warp(
            frames_src_sh.reshape(batch * num_frames, channel, height_sh, width_sh),
            M,
            dsize=(height_sh, width_sh),
        ).reshape_as(frames_src_sh)

        # loss
        loss_reconst = loss_func(src_warped, frames_dst_sh)[frames_dst_mask_sh].mean()
        if smoothness_order == 0:
            loss_regularize = delta_constrained.abs().mean()
        elif smoothness_order == 1:
            loss_regularize = (delta_constrained[:, 1:] - delta_constrained[:, :-1]).abs().mean()
        elif smoothness_order == 2:
            loss_regularize = (delta_constrained[:, 2:] - 2 * delta_constrained[:, 1:-1] + delta_constrained[:, :-2]).abs().mean()
        else:
            raise NotImplementedError(f"{smoothness_order=} unsupported.")

        # update
        loss = loss_reconst + loss_regularize * smoothness_weight
        loss.backward()
        if fix_first_frame:
            delta.grad[:, 0] = 0.0  # Zero grad for 0-th frame
        optimizer.step()

        # if iter % 100 == 0 or iter == max_iters - 1:
        #     print(f"[homography_estimation] {iter=}, loss_reconst={loss_reconst.item()}, loss_regularize={loss_regularize.item()}")

    with torch.no_grad():
        # final homography
        delta_constrained = torch.tanh(delta) * corner_max_shift_ratio * 2
        corners_tgt = corners_src + delta_constrained
        M = get_perspective_transform(corners_src.reshape(-1, 4, 2), corners_tgt.reshape(-1, 4, 2))

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
