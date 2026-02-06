from typing import Literal
import math

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
        padding_noise_strength: float = 0.0,
        init_homography: Float[torch.Tensor, "batch num_frames 3 3"] | None = None,  # src -> dst
        invert_output_homography: bool = False,  # if True, returns dst -> src
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

    if init_homography is None:
        raise NotImplementedError("init_homography is required.")
    assert init_homography.shape == (batch, num_frames, 3, 3)
    assert init_homography[..., 2, 2].abs().min() > 1e-3, f"init_homography[..., 2, 2] is close to 0: {init_homography=}"
    init_homography = init_homography / init_homography[..., 2:3, 2:3]

    # NOTE: TESTING SINGLE PARAM (num_frames -> 1)
    alpha = torch.nn.Parameter(torch.full((batch, 1, 1, 1), -3.0, device=frames_src.device, dtype=frames_src.dtype))
    # optimizer = torch.optim.Adam([alpha], lr=lr)
    optimizer = torch.optim.LBFGS([alpha], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def zero_first_frame_grad():
        if fix_first_frame and alpha.grad is not None and alpha.shape[1] > 1:
            alpha.grad[:, 0] = 0.0

    def compute_loss():
        optimizer.zero_grad()

        # linear interpolation between init_homography and identity
        ratio = torch.sigmoid(alpha)
        M_identity = torch.eye(3, device=frames_src.device, dtype=frames_src.dtype).reshape(1, 1, 3, 3)
        M = ratio * init_homography + (1 - ratio) * M_identity

        # warp
        src_warped = homography_warp(
            frames_src_sh.reshape(batch * num_frames, channel, height_sh, width_sh),
            torch.linalg.inv(M).reshape(batch * num_frames, 3, 3),  # NOTE: homography_warp expects dst->src
            dsize=(height_sh, width_sh),
        ).reshape_as(frames_src_sh)

        # loss
        loss_reconst = loss_func(src_warped, frames_dst_sh)[frames_dst_mask_sh].mean()
        if smoothness_order < 0:
            loss_regularize = 0.0
        elif smoothness_order == 0:
            loss_regularize = ratio.abs().mean()
        elif smoothness_order == 1:
            assert ratio.shape[1] >= 2, f"{ratio.shape=}"
            loss_regularize = (ratio[:, 1:] - ratio[:, :-1]).abs().mean()
        elif smoothness_order == 2:
            assert ratio.shape[1] >= 3, f"{ratio.shape=}"
            loss_regularize = (ratio[:, 2:] - 2 * ratio[:, 1:-1] + ratio[:, :-2]).abs().mean()
        else:
            raise NotImplementedError(f"{smoothness_order=} unsupported.")

        # update
        loss = loss_reconst + loss_regularize * smoothness_weight
        loss.backward()
        zero_first_frame_grad()
        return loss


    for iter in range(max_iters):
        #_ = compute_loss()
        optimizer.step(compute_loss)

    with torch.no_grad():
        # final homography
        print(f"{alpha.flatten()=}")
        ratio = torch.sigmoid(alpha)
        M = ratio * init_homography + (1 - ratio) * torch.eye(3, device=frames_src.device, dtype=frames_src.dtype).reshape(1, 1, 3, 3)
        M_inv = torch.linalg.inv(M)

        # apply the optimized warp
        src_warped = homography_warp(
            frames_src.flatten(0, 1),
            M_inv.flatten(0, 1),  # NOTE: homography_warp expects dst->src
            dsize=(height, width),
            padding_mode=padding_mode,
        ).reshape_as(frames_src)

        if padding_noise_strength > 0:
            assert padding_noise_strength <= 1.0, f"{padding_noise_strength=} must be within [0, 1]"
            valid_region = homography_warp(
                torch.ones(batch * num_frames, 1, height, width, dtype=frames_src.dtype, device=frames_src.device),
                M_inv.flatten(0, 1),  # NOTE: homography_warp expects dst->src
                dsize=(height, width),
                padding_mode='zeros',
            ).reshape(batch, num_frames, 1, height, width)
            padding_mask = (valid_region < 0.5).expand_as(src_warped)
            padding_pixel_num = padding_mask.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)

            # variance in padded region
            src_warped_mean = (src_warped * padding_mask).sum(dim=(2, 3, 4), keepdim=True) / padding_pixel_num
            src_warped_diff = (src_warped - src_warped_mean) * padding_mask
            src_warped_var = (src_warped_diff ** 2).sum(dim=(2, 3, 4), keepdim=True) / (padding_pixel_num - 1).clamp(min=1.0)
            src_warped_var *= padding_pixel_num > 1

            # add noise while keeping variance
            noise = torch.randn_like(src_warped)
            src_warped_all_noised = math.sqrt(1 - padding_noise_strength) * src_warped + \
                                    torch.sqrt(padding_noise_strength * src_warped_var) * noise
            src_warped[padding_mask] = src_warped_all_noised[padding_mask]

    if invert_output_homography:
        return M_inv, src_warped
    else:
        return torch.linalg.inv(M_inv), src_warped
