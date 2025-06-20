from typing import Callable, Dict, List, Optional, Union
import argparse
import inspect
import os
from dataclasses import dataclass
from tqdm import tqdm

import lovely_tensors
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils import BaseOutput, logging, load_image, export_to_video
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines import DiffusionPipeline
from einops import rearrange
from jaxtyping import Float
from kornia.filters import box_blur
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from scheduling_euler_discrete import EulerDiscreteScheduler
from unet import MyUNet
from warp import homography_estimation

lovely_tensors.monkey_patch()

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def box_blur_3D(
        tensor: Float[torch.Tensor, "b f c h w"],
        s_kernel_size: int,
        t_kernel_size: int,
        border_type: str = 'reflect',
    ) -> Float[torch.Tensor, "b f c h w"]:
    """Helper to compute the mean over a 3D spatio-temporal-channel window."""
    assert s_kernel_size % 2 == 1
    assert t_kernel_size % 2 == 1
    k_temporal = t_kernel_size // 2
    b, f, c, h, w = tensor.shape

    # 1. Temporal mean pooling using 1D convolution
    temp_tensor = rearrange(tensor, "b f c h w ->  (b c h w) () f")
    temporal_kernel = torch.ones(1, 1, t_kernel_size, device=tensor.device, dtype=tensor.dtype) / t_kernel_size
    padded_tensor = F.pad(temp_tensor, (k_temporal, k_temporal), mode=border_type)
    temp_mean = F.conv1d(padded_tensor, temporal_kernel, padding='valid')
    temp_mean = rearrange(temp_mean, "(b c h w) () f -> b f c h w", b=b, f=f, c=c, h=h, w=w)

    # 2. Spatial mean pooling using kornia's box_blur
    spatio_temporal_mean = box_blur(
        rearrange(temp_mean, "b f c h w -> (b f) c h w"),
        (s_kernel_size, s_kernel_size),
        border_type='reflect',
    ).reshape_as(temp_mean)

    return spatio_temporal_mean


def safe_division_2D(
        nunom: Float[torch.Tensor, "b c h w"],
        denom: Float[torch.Tensor, "b c h w"],
        kernel_size: int = 1,
        eps: float = 1e-12,
    ) -> Float[torch.Tensor, "b c h w"]:
    """
    Return A/B, assuming that A/B is theoretically a continuous finite function.
    The solution R is derived as R(p) = argmin_r Σ_{q in W(p)} (A(q) - r * B(q))^2
    It has an analytical solution: R(p) = (Σ_{q in W(p)} A(q)B(q)) / (Σ_{q in W(p)} B(q)^2)
    """
    kernel_size_tupled = (kernel_size * 2 + 1, kernel_size * 2 + 1)
    sum_numom_denom = box_blur(nunom * denom, kernel_size_tupled)
    sum_denom_squared = box_blur(denom * denom, kernel_size_tupled)
    return sum_numom_denom / (sum_denom_squared + eps)


def safe_division_3D(
        nunom: Float[torch.Tensor, "b f c h w"],
        denom: Float[torch.Tensor, "b f c h w"],
        k_spatial: int = 1,
        k_temporal: int = 1,
        eps: float = 1e-12,
    ) -> Float[torch.Tensor, "b c h w"]:
    """
    Return A/B, assuming that A/B is theoretically a continuous finite function.
    The solution R is derived as R(p) = argmin_r Σ_{q in W(p)} (A(q) - r * B(q))^2
    It has an analytical solution: R(p) = (Σ_{q in W(p)} A(q)B(q)) / (Σ_{q in W(p)} B(q)^2)
    """
    s_kernel_size = 2 * k_spatial + 1
    t_kernel_size = 2 * k_temporal + 1
    sum_numom_denom = box_blur_3D(nunom * denom, s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size)
    sum_denom_squared = box_blur_3D(denom * denom, s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size)
    return sum_numom_denom / (sum_denom_squared + eps)


def local_covariance_2D(
        X: Float[torch.Tensor, "b f c h w"],
        Y: Float[torch.Tensor, "b f c h w"],
        k: int,
    ) -> Float[torch.Tensor, "b f 1 h w"]:
    """
    Calculates the local covariance map between two tensors over a sliding window.

    Args:
        X (torch.Tensor): The first tensor, shape (B, F, C, H, W).
        Y (torch.Tensor): The second tensor, shape (B, F, C, H, W).
        k (int): The radius of the square window. The total window size will be (2k+1)x(2k+1).

    Returns:
        torch.Tensor: A 2D tensor of shape (H, W) where each element (i, j) is the
                      covariance between the windows of the input tensors centered at (i, j).
    """
    if X.shape != Y.shape:
        raise ValueError("Input tensors must have the same shape.")
    B, num_frames, C, H, W = X.shape
    kernel_size = 2 * k + 1

    # The number of elements in each window for a single frame calculation.
    # This is over channels and the spatial window.
    n_elements_in_window = C * (kernel_size ** 2)

    # Reshape to treat each frame as an independent item in a large batch.
    # Shape changes from (B, F, C, H, W) -> (B * F, C, H, W)
    X_reshaped = X.reshape(B * num_frames, C, H, W)
    Y_reshaped = Y.reshape(B * num_frames, C, H, W)

    # --- Step 1: Calculate E[X] and E[Y] for each local window ---
    # `F.avg_pool2d` operates on the (B*F) batch, calculating the spatial mean
    # for each of the C channels independently.
    pool_latents = F.avg_pool2d(X_reshaped, kernel_size=kernel_size, stride=1, padding=k, count_include_pad=False)
    pool_pseudo = F.avg_pool2d(Y_reshaped, kernel_size=kernel_size, stride=1, padding=k, count_include_pad=False)
    # The shape of pool_latents and pool_pseudo is (B*F, C, H, W).

    # To get the mean over the window (channels + spatial), we average over the channel dim (dim=1).
    local_mean_latents = torch.mean(pool_latents, dim=1) # Shape: (B*F, H, W)
    local_mean_pseudo = torch.mean(pool_pseudo, dim=1)  # Shape: (B*F, H, W)

    # --- Step 2: Calculate E[X * Y] for each local window ---
    product_tensor = X_reshaped * Y_reshaped # Shape: (B*F, C, H, W)
    pool_product = F.avg_pool2d(product_tensor, kernel_size=kernel_size, stride=1, padding=k, count_include_pad=False)
    local_mean_product = torch.mean(pool_product, dim=1) # Shape: (B*F, H, W)

    # --- Step 3: Apply the population covariance formula ---
    local_cov_map = local_mean_product - (local_mean_latents * local_mean_pseudo)

    # --- Step 4: Apply unbiased correction factor ---
    if n_elements_in_window > 1:
        correction_factor = n_elements_in_window / (n_elements_in_window - 1)
        local_cov_map_unbiased = local_cov_map * correction_factor
    else:
        local_cov_map_unbiased = local_cov_map

    # --- Step 5: Reshape the output to the desired (B, F, 1, H, W) shape ---
    final_cov_map = local_cov_map_unbiased.view(B, num_frames, 1, H, W)

    return final_cov_map


def local_covariance_3D(
        X: Float[torch.Tensor, "b f c h w"],
        Y: Float[torch.Tensor, "b f c h w"],
        k_spatial: int,
        k_temporal: int,
    ) -> Float[torch.Tensor, "b f 1 h w"]:
    """
    Calculates the local covariance map between two tensors over a sliding window.

    Args:
        X (torch.Tensor): The first tensor, shape (B, F, C, H, W).
        Y (torch.Tensor): The second tensor, shape (B, F, C, H, W).
        k_spatial (int): The radius of the spatial square window.
        k_temporal (int): The radius of the temporal square window.

    Returns:
        torch.Tensor: A tensor of shape (B, F, 1, H, W) where each element (b, f, 0, i, j) is the
                      covariance between the windows of the input tensors centered at (i, j).
    """
    if X.shape != Y.shape:
        raise ValueError("Input tensors must have the same shape (B, F, C, H, W)")
    if k_spatial < 0 or k_temporal < 0:
        raise ValueError("Window radii k_spatial and k_temporal must be non-negative.")

    _b, _f, C, _h, _w = X.shape
    s_kernel_size = 2 * k_spatial + 1
    t_kernel_size = 2 * k_temporal + 1

    # Calculate E[X], E[Y], and E[X*Y] using the 3D mean pool helper
    mean_x = box_blur_3D(X, s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size)
    mean_y = box_blur_3D(Y, s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size)
    mean_xy = box_blur_3D(X * Y, s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size)

    # Channel mean
    mean_x = torch.mean(mean_x, dim=2, keepdim=True)  # Shape: (B, F, 1, H, W)
    mean_y = torch.mean(mean_y, dim=2, keepdim=True)  # Shape: (B, F, 1, H, W)
    mean_xy = torch.mean(mean_xy, dim=2, keepdim=True)  # Shape: (B, F, 1, H, W)

    # Apply the covariance formula: Cov(X, Y) = E[X*Y] - E[X]*E[Y]
    covariance = mean_xy - mean_x * mean_y

    # The number of elements in the 3D window (spatial, temporal, and channel dimensions).
    n_elements_in_window = C * (s_kernel_size ** 2) * t_kernel_size

    # Apply the unbiased sample covariance correction factor: N / (N-1)
    if n_elements_in_window > 1:
        correction_factor = n_elements_in_window / (n_elements_in_window - 1)
        covariance = covariance * correction_factor

    return covariance



def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


################################################################


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0


            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14, permute: bool = True):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        if permute:
            frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        else:
            frames = frames.reshape(-1, num_frames, *frames.shape[1:])

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # @torch.no_grad()
    def __call__(
        self,
        enable_nvssolver: bool,
        warped_images: List[PIL.Image.Image],
        warped_masks: List[PIL.Image.Image],
        denoise_start_step: Optional[int],
        repaint_iter_num: int,
        lambda_ts,
        lr: float,
        weight_clamp: float,
        ########
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.0,  # NOTE: Modified
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(warped_images, height, width)
        if denoise_start_step is not None:
            assert denoise_start_step < num_inference_steps

        # 1.5 preprocess: inpaint the holes with the mean color
        assert len(warped_images) == len(warped_masks)
        print("Warning: The void regions in the warped images are filled with the mean color of the non-void regions.")
        for i, (img, msk) in enumerate(zip(warped_images, warped_masks)):
            img = np.array(img)
            msk = np.array(msk).mean(axis=-1)
            img[msk >= 128] = np.mean(img[msk < 128], axis=0)
            warped_images[i] = PIL.Image.fromarray(img)

        # 2. Define call parameters
        batch_size = 1  # NOTE: Modified
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        with torch.no_grad():
            image_embeddings = self._encode_image(warped_images[0], device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        warped_images = self.video_processor.preprocess(warped_images, height=height, width=width).to(device)
        noise = randn_tensor(warped_images.shape, generator=generator, device=warped_images.device, dtype=warped_images.dtype)
        warped_images = warped_images + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            warped_latents = []
            for img in warped_images:
                warped_latent_ =  self._encode_vae_image(
                    img.unsqueeze(0),
                    device=device,
                    num_videos_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                ) # [2, 4, 72, 128]
                warped_latents.append(warped_latent_.unsqueeze(1))

        warped_latents = torch.cat(warped_latents, dim=1).to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = warped_latents[:, 0:1].repeat(1, num_frames, 1, 1, 1)

        # 4.5 Prepare pixel-space and latent-space masks
        warped_masks = [torch.from_numpy(np.array(msk).mean(axis=-1) > 128).float() for msk in warped_masks]
        warped_masks = torch.stack(warped_masks, dim=0).to(device)
        warped_masks = rearrange(warped_masks, "f h w -> () f () h w")
        warped_masks_sh = rearrange(warped_masks, "() f () (nh ph) (nw pw) -> () f () nh nw (ph pw)", ph=8, pw=8)
        warped_masks_sh = warped_masks_sh.mean(dim=-1)
        warped_masks_sh = torch.clip(warped_masks_sh * 5, 0, 1)

        # os.makedirs("dump", exist_ok=True)
        # torch.save(warped_latents, "dump/warped_latents.pt")
        # torch.save(warped_masks_sh, "dump/warped_masks_sh.pt")

        # To use warped_latents inside the denoising loop, it must be scaled!!!
        # NOTE: The VAE scaling is unnecessary for image_latents
        warped_latents = warped_latents * self.vae.config.scaling_factor

        # For later convenience
        warped_images = rearrange(warped_images, "f c h w -> () f c h w")

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        if denoise_start_step is not None:
            if latents is not None:
                print("Warning: when denoise_start_step is set, 'latents' argument is ignored.")
            warped_latents_per_batch = warped_latents[batch_size:] if self.do_classifier_free_guidance else warped_latents
            noise = randn_tensor(warped_latents_per_batch.shape, generator=generator, device=device, dtype=warped_latents_per_batch.dtype)
            # NOTE: sigmas are accessible after scheduler.set_timesteps is called
            latents = warped_latents_per_batch + noise * self.scheduler.sigmas[denoise_start_step]
        else:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if (denoise_start_step is not None) and i < denoise_start_step:
                    progress_bar.update()
                    continue

                if enable_nvssolver:
                    grads = []
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t,step_i=i)
                    latent_model_input = torch.cat([latent_model_input[0:1], image_latents[1:2]], dim=2)
                    for ii in range(4):
                        with torch.enable_grad():

                            latents.requires_grad_(True)
                            latents.retain_grad()
                            image_latents.requires_grad_(True)
                            latent_model_input = latent_model_input.detach()
                            latent_model_input.requires_grad = True
                            # print('latent_model_input',latent_model_input.shape)

                            named_param = list(self.unet.named_parameters())
                            for n,p in named_param:
                                p.requires_grad = False
                            if ii == 0:
                                latent_model_input1 = latent_model_input[0:1,:,:,:40,:72]
                                latents1 = latents[0:1,:,:,:40,:72]
                                warped_latents1 = warped_latents[:2,:,:,:40,:72]
                                mask1 = warped_masks_sh[0:1,:,:,:40,:72]
                            elif ii ==1:
                                latent_model_input1 = latent_model_input[0:1,:,:,32:,:72]
                                latents1 = latents[0:1,:,:,32:,:72]
                                warped_latents1 = warped_latents[:2,:,:,32:,:72]
                                mask1 = warped_masks_sh[0:1,:,:,32:,:72]
                            elif ii ==2:
                                latent_model_input1 = latent_model_input[0:1,:,:,:40,56:]
                                latents1 = latents[0:1,:,:,:40,56:]
                                warped_latents1 = warped_latents[:2,:,:,:40,56:]
                                mask1 = warped_masks_sh[0:1,:,:,:40,56:]
                            elif ii ==3:
                                latent_model_input1 = latent_model_input[0:1,:,:,32:,56:]
                                latents1 = latents[0:1,:,:,32:,56:]
                                warped_latents1 = warped_latents[:2,:,:,32:,56:]
                                mask1 = warped_masks_sh[0:1,:,:,32:,56:]
                            image_embeddings1 = image_embeddings[0:1,:,:]
                            added_time_ids1 =added_time_ids[0:1,:]
                            torch.cuda.empty_cache()
                            noise_pred_t = self.unet(
                                latent_model_input1,
                                t,
                                encoder_hidden_states=image_embeddings1,
                                added_time_ids=added_time_ids1,
                                return_dict=False,
                            )[0]

                            output = self.scheduler.step_single(
                                noise_pred_t,
                                t,
                                latents1,
                                warped_latents1,
                                mask1,
                                lambda_ts,
                                step_i=i,
                                lr=lr,
                                weight_clamp=weight_clamp,
                                compute_grad=True)
                            grad = output.grad
                            grads.append(grad)

                    grads1 = torch.cat((grads[0],grads[1][:,:,:,8:,:]),-2)
                    grads2 = torch.cat((grads[2],grads[3][:,:,:,8:,:]),-2)
                    grads3 = torch.cat((grads1,grads2[:,:,:,:,16:]),-1)
                    latents = latents - grads3.half()





                with torch.no_grad():
                    for j in range(repaint_iter_num):
                        latents_ori = latents.clone()

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)

                        # Concatenate image_latents over channels dimension
                        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                        # predict the noise residual
                        if i < self._num_timesteps * 2 // 3:
                            self.unet.latent_shape_ = (
                                1,  # 2 if self.do_classifier_free_guidance else 1,
                                num_frames, 8, height//8, width//8)  # NOTE: image_latents is appended so the channel num is 8

                            self.unet.inject(None)
                            noise_pred_uncond = self.unet(
                                latent_model_input.split(batch_size, dim=0)[i%2],
                                t,
                                encoder_hidden_states=image_embeddings.split(batch_size, dim=0)[i%2],
                                added_time_ids=added_time_ids.split(batch_size, dim=0)[i%2],
                                return_dict=False,
                                record_attention=False,
                            )[0]

                            self.unet.inject(warped_masks_sh < 0.5)
                            noise_pred_cond = self.unet(
                                latent_model_input[batch_size:],
                                t,
                                encoder_hidden_states=image_embeddings[batch_size:],
                                added_time_ids=added_time_ids[batch_size:],
                                return_dict=False,
                            )[0]

                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        else:
                            self.unet.latent_shape_ = (
                                2 if self.do_classifier_free_guidance else 1,
                                num_frames, 8, height//8, width//8)  # NOTE: image_latents is appended so the channel num is 8

                            self.unet.inject(None)
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=image_embeddings,
                                added_time_ids=added_time_ids,
                                return_dict=False,
                            )[0]

                            # perform guidance
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        out = self.scheduler.step_single(noise_pred, t, latents, None, None, lambda_ts, step_i=i, compute_grad=False)
                        pseudo_x0 = out.pred_original_sample
                        latents = out.prev_sample

                        # alignment
                        if i < self._num_timesteps // 2:
                            M, pseudo_x0 = homography_estimation(
                                pseudo_x0,
                                warped_latents[batch_size:] if self.do_classifier_free_guidance else warped_latents,
                                warped_masks_sh < 0.5,
                                process_size=128,
                                lr=1e-2,
                                max_iters=100,
                                num_control_points=num_frames//3,
                                fix_first_frame=True,
                                acceleration_penalty_weight=0.5,
                                padding_mode="border",
                            )
                            from kornia.geometry.transform.imgwarp import homography_warp
                            latents = homography_warp(
                                latents.flatten(0, 1).float(),
                                M,
                                dsize=latents.shape[-2:],
                                padding_mode="reflection",
                            ).reshape_as(latents).to(latents.dtype)

                        # resampling
                        if j < repaint_iter_num - 1:

                            sigma_t = self.scheduler.sigmas[i]
                            var_data = warped_latents[batch_size:, 0:1].var() * 0.5

                            # os.makedirs("dump", exist_ok=True)
                            # torch.save(latents_ori, f"dump/latents_ori_{i}_{j}.pt")
                            # torch.save(pseudo_x0, f"dump/pseudo_x0_{i}_{j}.pt")

                            if i < self._num_timesteps * 1 // 2:
                                sigma_s = self.scheduler.sigmas[i+1]
                            else:
                                # deduce optimal sigma_s for RePaint
                                k_spatial = 5
                                k_temporal = 3
                                var_latents_ori = local_covariance_3D(latents_ori, latents_ori, k_spatial, k_temporal)
                                var_pseudo_x0 = local_covariance_3D(pseudo_x0, pseudo_x0, k_spatial, k_temporal)
                                cov_latents_ori_pseudo_x0 = local_covariance_3D(latents_ori, pseudo_x0, k_spatial, k_temporal)
                                coeff_A = var_latents_ori - 2 * cov_latents_ori_pseudo_x0 + var_pseudo_x0 - sigma_t**2
                                coeff_B = (var_pseudo_x0 - cov_latents_ori_pseudo_x0) * sigma_t
                                coeff_C = (var_pseudo_x0 - var_data) * sigma_t**2
                                nunom = coeff_B - torch.sqrt(torch.relu(coeff_B.pow(2) - coeff_A * coeff_C))
                                sigma_s = safe_division_3D(nunom, coeff_A, k_spatial, k_temporal)
                                sigma_s = torch.clamp(sigma_s, 0, sigma_t)
                                # print(f"{var_latents_ori=}\n{var_pseudo_x0=}\n{cov_latents_ori_pseudo_x0=}\n{coeff_A=}\n{coeff_B=}\n{coeff_C=}\n{nunom=}\n{sigma_s=}")

                                # zero out sigma_s where we have valid masks
                                # from kornia.contrib import distance_transform
                                # distance_thresh = 7
                                # dt = distance_transform(1 - rearrange(warped_masks_sh, "b f c h w -> (b f) c h w")).reshape_as(warped_masks_sh)
                                # dt = torch.clamp(dt / distance_thresh, 0, 1)
                                # sigma_s = sigma_s * dt
                                # sigma_s = torch.clamp(sigma_s, 0, sigma_t)

                                # torch.save(sigma_s, f"dump/sigma_s_{i}_{j}.pt")

                                print(f"{sigma_s=}, {sigma_t=}")


                            # direct pasting
                            latents_mid = (sigma_s / sigma_t) * latents_ori + (1 - sigma_s / sigma_t) * pseudo_x0
                            warped_latents_noisy = warped_latents + sigma_s * randn_tensor(
                                warped_latents.shape, generator=generator, device=warped_latents.device, dtype=warped_latents.dtype)
                            latents_mid_pasted = warped_masks_sh * latents_mid + \
                                        (1 - warped_masks_sh) * (warped_latents_noisy[batch_size:] if self.do_classifier_free_guidance else warped_latents_noisy)

                            opt_std = 0.5
                            posterior_sigma = sigma_t**2 / opt_std
                            latents = self.stochastic_resample(
                                opt_zs=latents_mid_pasted,
                                ori_zt=latents_ori,
                                sigma_s=sigma_s,
                                sigma_t=sigma_t,
                                posterior_sigma=posterior_sigma,
                                generator=generator,
                            )


                            print(f"{latents_ori.var().item()=}, {(var_data + sigma_t**2).item()=}")
                            print(f"{pseudo_x0.var().item()=}, {var_data.item()=}")
                            print(f"{latents_mid.var().item()=}, var_data + sigma_s**2={var_data + (sigma_s.mean() if isinstance(sigma_s, torch.Tensor) else sigma_s)**2}")
                            print(f"{latents_mid_pasted.var().item()=}")
                            print(f"{latents.var().item()=}, {(var_data + sigma_t**2).item()=}")
                            print()


                        else:
                            pass


                        latents = latents.half()

                    del latents_ori

                    torch.cuda.empty_cache()


                # os.makedirs("dump", exist_ok=True)
                # torch.save(pseudo_x0, f"dump/latents_with_resample_{i}.pt")


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)



                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            with torch.no_grad():
                frames = self.decode_latents(latents, num_frames, decode_chunk_size)
                frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


    def stochastic_resample(
            self,
            opt_zs: torch.Tensor,
            ori_zt: Optional[torch.Tensor],
            sigma_s: float | torch.Tensor,
            sigma_t: float,
            posterior_sigma: float = torch.inf,
            generator: Optional[torch.Generator] = None,
        ):
        """
        Function to resample z_t based on the ReSample paper.
        The formulation is translated from VP to VE to adapt to SVD.

        Arguments:
            opt_zs: hat{z}_s(y)
            ori_zt: z'_t
            sigma_s: The noise level of opt_zs.
            sigma_t: The noise level of ori_zt.
            posterior_sigma: p(z'_t | hat{z}_t, hat{z}_s, y) ~ N(hat{z}_s(y), posterior_sigma)
        """
        if isinstance(sigma_s, torch.Tensor):
            assert torch.all(0 <= sigma_s) and torch.all(sigma_s <= sigma_t), f"{sigma_s.min()=}, {sigma_s.max()=}"
        else:
            assert 0 <= sigma_s <= sigma_t, f"{sigma_s=}, {sigma_t=}"
        noise = randn_tensor(opt_zs.shape, generator=generator, device=opt_zs.device, dtype=opt_zs.dtype)

        t_squared_minus_s_squared = sigma_t ** 2 - sigma_s ** 2
        post_sigma_squared = posterior_sigma ** 2

        if posterior_sigma == torch.inf:
            return opt_zs + noise * t_squared_minus_s_squared**0.5
        else:
            assert ori_zt is not None

        denom = post_sigma_squared + t_squared_minus_s_squared
        mean = (post_sigma_squared / denom) * opt_zs + (t_squared_minus_s_squared / denom) * ori_zt
        std = posterior_sigma * torch.sqrt(t_squared_minus_s_squared / (post_sigma_squared + t_squared_minus_s_squared))
        return mean + noise * std


def search_hypers(sigmas, num_frames: int):
    sigmas = sigmas[:-1]
    sigmas_max = max(sigmas)

    v2_list = np.arange(50, 1001, 50)
    v3_list = np.arange(10, 101, 10)
    v1_list = np.linspace(0.001, 0.009, 9)
    zero_count_default = 0
    index_list = list(range(1, num_frames))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    for v1 in v1_list:
        for v2 in v2_list:
            for v3 in v3_list:
                flag = True
                lambda_t_list = []
                for sigma in sigmas:
                    sigma_n = sigma/sigmas_max
                    temp_cond_indices = [0]
                    for tau in range(25):
                        if tau not in index_list:
                            lambda_t_list.append(1)
                        else:
                            tau_p = 0
                            tau_ = tau/24

                            Q = v3 * abs((tau_-tau_p)) - v2*sigma_n
                            k = 0.8
                            b = -0.2

                            lambda_t_1 = (-(2*v1 + k*Q) + ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            lambda_t_2 = (-(2*v1 + k*Q) - ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            v1_ = -v1
                            lambda_t_3 = (-(2*v1_ + k*Q) + ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            lambda_t_4 = (-(2*v1_ + k*Q) - ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            try:
                                if np.isreal(lambda_t_1):
                                    if lambda_t_1 >1.0:
                                        lambda_t = lambda_t_1
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_2):
                                    if lambda_t_2 >1.0:
                                        lambda_t = lambda_t_2
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_3):
                                    if lambda_t_3 <=1.0 and lambda_t_3>0:
                                        lambda_t = lambda_t_3
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_4):
                                    if lambda_t_4 <=1.0 and lambda_t_4>0:
                                        lambda_t = lambda_t_4
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                flag = False
                                break
                            except:
                                flag = False
                                break
                            lambda_t_list.append(lambda_t/(1+lambda_t))


                if flag == True:
                    zero_count = sum(1 for x in lambda_t_list if x > 0.5)
                    if zero_count > zero_count_default:
                        zero_count_default = zero_count
                        v_optimized = [v1,v2,v3]
                        lambda_t_list_optimized = lambda_t_list

    lambda_t_list_optimized = np.array(lambda_t_list_optimized)
    lambda_t_list_optimized = lambda_t_list_optimized.reshape([len(sigmas), num_frames])

    return lambda_t_list_optimized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_folder",
        type=str,
    )

    parser.add_argument(
        "--trajectory_folder",
        type=str,
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100
    )

    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--enable_nvssolver",
        action="store_true",
    )

    parser.add_argument(
        "--denoise_start_step",
        type=int,
        default=None,
        help="If you enable resample, num_inference_steps // 3 is the recommended value"
    )

    parser.add_argument(
        "--repaint_iter_num",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--weight_clamp",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # load pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.unet = MyUNet.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # calculate lambda
    sigma_list = np.load(f'sigmas/sigmas_{args.num_inference_steps}.npy').tolist()
    lambda_ts = search_hypers(sigma_list, args.num_frames) if args.enable_nvssolver else []
    lambda_ts = torch.tensor(lambda_ts)

    # load images
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    warped_masks = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}_mask.png")) for i in range(args.num_frames)]

    # inference
    svd_output = pipe(
        enable_nvssolver=args.enable_nvssolver,
        warped_images=warped_images,
        warped_masks=warped_masks,
        denoise_start_step=args.denoise_start_step,  # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        repaint_iter_num=args.repaint_iter_num,  # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ########
        lambda_ts=lambda_ts,
        lr=args.lr,
        weight_clamp=args.weight_clamp,
        num_frames=args.num_frames,
        decode_chunk_size=8,
        num_inference_steps=args.num_inference_steps,
        min_guidance_scale=args.min_guidance_scale,
        max_guidance_scale=args.max_guidance_scale,
        generator=torch.manual_seed(args.seed),
    )
    frames = svd_output.frames[0]

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
