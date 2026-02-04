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
        constrain_to_init_line: bool = True,
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

    # base corners
    corners_src = torch.tensor([
        [-1.0, -1.0],  # top-left
        [ 1.0, -1.0],  # top-right
        [ 1.0,  1.0],  # bottom-right
        [-1.0,  1.0],  # bottom-left
    ], device=frames_src.device, dtype=frames_src.dtype).reshape(1, 1, 4, 2).expand(batch, num_frames, 4, 2)

    alpha = None
    delta = None
    compute_delta = None

    if constrain_to_init_line:
        if init_homography is None:
            raise NotImplementedError("constrain_to_init_line=True requires init_homography.")

        corners_src_hom = torch.cat([corners_src, torch.ones_like(corners_src[:, :, :, 0:1])], dim=-1)
        corners_ini_hom = torch.matmul(corners_src_hom, init_homography.mT)      # (batch, num_frames, 4, 3)
        assert torch.all(corners_ini_hom[:, :, :, 2:3].abs()) > 1e-3, "init_homography is not valid."
        corners_ini = corners_ini_hom[:, :, :, 0:2] / corners_ini_hom[:, :, :, 2:3]  # (batch, num_frames, 4, 2)
        direction = corners_ini - corners_src

        # NOTE: TESTING SINGLE PARAM (num_frames -> 1)
        alpha = torch.nn.Parameter(torch.full((batch, 1, 1, 1), 0.0, device=frames_src.device, dtype=frames_src.dtype))
        # optimizer = torch.optim.Adam([alpha], lr=lr)
        optimizer = torch.optim.LBFGS([alpha], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        def compute_delta():
            return torch.sigmoid(alpha) * direction

        def zero_first_frame_grad():
            if fix_first_frame and alpha.grad is not None and alpha.shape[1] > 1:
                alpha.grad[:, 0] = 0.0

    else:
        if init_homography is not None:
            corners_src_hom = torch.cat([corners_src, torch.ones_like(corners_src[:, :, :, 0:1])], dim=-1)
            corners_ini_hom = torch.matmul(corners_src_hom, init_homography.mT)      # (batch, num_frames, 4, 3)
            assert torch.all(corners_ini_hom[:, :, :, 2:3].abs()) > 1e-3, "init_homography is not valid."
            corners_ini = corners_ini_hom[:, :, :, 0:2] / corners_ini_hom[:, :, :, 2:3]  # (batch, num_frames, 4, 2)
            delta_constrained = corners_ini - corners_src

            delta = torch.nn.Parameter(torch.zeros(batch, num_frames, 4, 2, device=frames_src.device, dtype=frames_src.dtype))
            delta_init = torch.atanh(delta_constrained / (corner_max_shift_ratio * 2))
            if not torch.isfinite(delta_init).all():
                print("[homography_estimation] homography init failed. Increase corner_max_shift_ratio or remove init_homography. Fallback to zero init.")
                delta_init = torch.zeros_like(delta_init)
            delta.data = delta_init
        else:
            delta = torch.nn.Parameter(torch.zeros(batch, num_frames, 4, 2, device=frames_src.device, dtype=frames_src.dtype))

        #optimizer = torch.optim.Adam([delta], lr=lr)
        optimizer = torch.optim.LBFGS([delta], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        def compute_delta():
            return torch.tanh(delta) * corner_max_shift_ratio * 2

        def zero_first_frame_grad():
            if fix_first_frame and delta.grad is not None:
                delta.grad[:, 0] = 0.0

    # blur setup
    from kornia.filters import GaussianBlur2d
    blur_layer = GaussianBlur2d(kernel_size=(9, 9), sigma=(1.0, 1.0))
    frames_dst_sh_blurred = blur_layer(frames_dst_sh.flatten(0, 1)).reshape_as(frames_dst_sh)

    def compute_loss():
        optimizer.zero_grad()

        # move corners
        delta_constrained = compute_delta()
        corners_tgt = corners_src + delta_constrained

        # deduce "inverse" homography (target -> source)
        M_inv = get_perspective_transform(corners_tgt.reshape(-1, 4, 2), corners_src.reshape(-1, 4, 2))

        # warp
        src_warped = homography_warp(
            frames_src_sh.reshape(batch * num_frames, channel, height_sh, width_sh),
            M_inv,  # NOTE: homography_warp expects dst->src
            dsize=(height_sh, width_sh),
        ).reshape_as(frames_src_sh)

        # blur src
        src_warped = blur_layer(src_warped.flatten(0, 1)).reshape_as(src_warped)

        # loss
        loss_reconst = loss_func(src_warped, frames_dst_sh_blurred)[frames_dst_mask_sh].mean()
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
        zero_first_frame_grad()
        return loss


    for iter in range(max_iters):
        #_ = compute_loss()
        optimizer.step(compute_loss)

    with torch.no_grad():
        # final homography
        delta_constrained = compute_delta()
        if constrain_to_init_line:
            print(f"{alpha.flatten()=}")

        corners_tgt = corners_src + delta_constrained
        M_inv = get_perspective_transform(corners_tgt.reshape(-1, 4, 2), corners_src.reshape(-1, 4, 2))
        M_inv = M_inv.reshape(batch, num_frames, 3, 3)

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
