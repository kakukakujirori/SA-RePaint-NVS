from typing import Callable, Dict, List, Optional, Union
import argparse
import inspect
import os
from dataclasses import dataclass
from tqdm import tqdm

import cv2
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
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from covariance import guided_blur_2D, local_covariance_2D, local_covariance_3D, safe_division_3D
from gpu_memory_monitor import GPUMemoryMonitor
from scheduling_euler_discrete import EulerDiscreteScheduler
from unet import MyUNet
from warp import homography_estimation

lovely_tensors.monkey_patch()

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# >>> Adaptive Projected Guidance (APG) >>>
class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def adaptive_projected_guidance(
        pred_uncond: torch.Tensor, # [B, F, C, H, W]
        pred_cond: torch.Tensor, # [B, F, C, H, W]
        guidance_scale: float,
        momentum_buffer: MomentumBuffer = None,
        eta: float = 0.5,
        norm_threshold: float = 400,
    ):

    def project(
        v0: torch.Tensor, # [B, F, C, H, W]
        v1: torch.Tensor, # [B, F, C, H, W]
    ):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
        print(f"[adaptive_projected_guidance] {norm_threshold=}, {diff_norm.item()=}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


momentum_buffer = MomentumBuffer(momentum=-0.0)

# <<< Adaptive Projected Guidance (APG) <<<


@torch.no_grad()
def get_var_data(
        q: Float[torch.Tensor, "num_frames num_heads hw c"],
        k: Float[torch.Tensor, "num_frames num_heads hw c"],
        warped_latents: Float[torch.Tensor, "batch num_frames c height width"],
        warped_masks_sh: Float[torch.Tensor, "batch num_frames () height width"],
        kernel_radius: int = 3,
        use_first_frame: bool = True,
        channelwise: bool = True,
    ) -> Float[torch.Tensor, "batch num_frames c^2 height width"]:
    batch, num_frames, _, height, width = warped_latents.shape
    scale_factor = int(round((height * width // q.shape[-2])**0.5))
    assert batch == 1
    assert height * width == q.shape[-2] * scale_factor**2, f"{height=}, {width=}, {q.shape=}, {scale_factor=}"
    assert num_frames == q.shape[0] == k.shape[0]
    h, w = height // scale_factor, width // scale_factor

    # local spatial variance of warped_masks_sh
    warped_variance = local_covariance_2D(warped_latents, warped_latents, k=kernel_radius, channelwise=channelwise)
    warped_variance_sh = F.interpolate(
        rearrange(warped_variance, "b f c2 h w -> (b f) c2 h w"),
        size=(h, w),
        mode="area",
    )
    v = rearrange(warped_variance_sh, "bf c2 h w -> bf () (h w) c2")
    v = v.expand(-1, k.shape[1], -1, -1)

    if use_first_frame:
        k = k[0:1, :, :, :].expand(num_frames, -1, -1, -1)
        v = v[0:1, :, :, :].expand(num_frames, -1, -1, -1)
        mask = None
    else:
        # attention masking
        mask = rearrange(warped_masks_sh < 0.5, "batch num_frames () height width -> (batch num_frames) () height width")
        mask = F.interpolate(mask.float(), size=(h, w), mode="area") > 0.5
        mask = rearrange(mask, "bf () h w -> bf () () (h w)").contiguous()

    # variance
    var_data_sh = F.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    var_data_sh = rearrange(var_data_sh, "num_frames num_heads (h w) c2 -> num_frames num_heads c2 h w", h=h, w=w)
    var_data_sh = var_data_sh.mean(dim=1)
    var_data = F.interpolate(
        var_data_sh,
        size=(warped_latents.shape[-2], warped_latents.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, num_frames, -1, height, width)

    return var_data


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
        warped_images: List[PIL.Image.Image],
        warped_masks: List[PIL.Image.Image],
        denoise_start_step: Optional[int],
        repaint_iter_num: int,
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
        print("Warning: The void regions in the warped images are filled by cv2.INPAINT_NS.")
        for i, (img, msk) in enumerate(zip(warped_images, warped_masks)):
            img = np.array(img)
            msk = np.array(msk).mean(axis=-1, keepdims=True).astype(np.uint8)
            img = cv2.inpaint(img, msk, 5, cv2.INPAINT_NS)
            warped_images[i] = PIL.Image.fromarray(img)
            # warped_images[i].save(f"inpainted_{i}.png")

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

        # To use warped_latents inside the denoising loop, it must be scaled!!!
        # NOTE: The VAE scaling is unnecessary for image_latents
        warped_latents = warped_latents * self.vae.config.scaling_factor

        # For later convenience
        warped_images = rearrange(warped_images, "f c h w -> () f c h w")

        # os.makedirs("dump", exist_ok=True)
        # torch.save(warped_latents, "dump/warped_latents.pt")
        # torch.save(warped_masks_sh, "dump/warped_masks_sh.pt")

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

                with torch.no_grad():
                    for j in range(repaint_iter_num):
                        latents_ori = latents.clone()

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)

                        # Concatenate image_latents over channels dimension
                        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                        # Attention weighting
                        base_weight = min(1, i / self._num_timesteps)
                        weight_map = (1 - warped_masks_sh) * (1 - base_weight) + base_weight

                        # negative
                        query_blur_sigma_invalid_max = 2  # TODO: MAGIC NUMBER!!!
                        query_blur_sigma_invalid_min = 2  # TODO: MAGIC NUMBER!!!
                        query_blur_sigma_invalid = query_blur_sigma_invalid_min + (query_blur_sigma_invalid_max - query_blur_sigma_invalid_min) * i / self._num_timesteps
                        query_blur_sigma_valid = 4  # TODO: MAGIC NUMBER!!!
                        query_blur_sigma = (1 - warped_masks_sh) * query_blur_sigma_valid + warped_masks_sh * query_blur_sigma_invalid

                        use_seg = i % 3 == 0

                        self.unet.latent_shape_ = (1, num_frames, 8, height//8, width//8)  # NOTE: image_latents is appended so the channel num is 8
                        self.unet.inject(weight_map, query_blur_sigma=query_blur_sigma if use_seg else None)
                        noise_pred_negative = self.unet(
                            latent_model_input.chunk(2)[1 if use_seg else 0],
                            t,
                            encoder_hidden_states=image_embeddings.chunk(2)[1 if use_seg else 0],
                            added_time_ids=added_time_ids.chunk(2)[1 if use_seg else 0],
                            return_dict=False,
                            record_attention=False,
                        )[0]

                        # positive
                        self.unet.latent_shape_ = (1, num_frames, 8, height//8, width//8)  # NOTE: image_latents is appended so the channel num is 8
                        self.unet.inject(weight_map)
                        noise_pred_positive = self.unet(
                            latent_model_input[batch_size:],
                            t,
                            encoder_hidden_states=image_embeddings[batch_size:],
                            added_time_ids=added_time_ids[batch_size:],
                            return_dict=False,
                            record_attention=True,
                        )[0]

                        # perform guidance
                        noise_pred = noise_pred_negative + self.guidance_scale * (noise_pred_positive - noise_pred_negative)
                        # noise_pred = adaptive_projected_guidance(
                        #     noise_pred_negative,
                        #     noise_pred_positive,
                        #     self.guidance_scale,
                        #     momentum_buffer,
                        # )

                        # retrieve qk
                        attn_query = self.unet.record_query_[0]
                        attn_key = self.unet.record_key_[0]
                        attn_query = attn_query[num_frames:] if attn_query.shape[0] > num_frames else attn_query
                        attn_key = attn_key[num_frames:] if attn_key.shape[0] > num_frames else attn_key
                        # os.makedirs("dump", exist_ok=True)
                        # torch.save(attn_query, f"dump/query_{i}_{j}_{0}.pt")
                        # torch.save(attn_key, f"dump/key_{i}_{j}_{0}.pt")

                        # compute the previous noisy sample x_t -> x_t-1
                        out = self.scheduler.step_single(noise_pred, t, latents, None, None, None, step_i=i, compute_grad=False)
                        pseudo_x0 = out.pred_original_sample
                        latents = out.prev_sample

                        if i >= self._num_timesteps * 4//5:
                            # This avoids flickering around small masks
                            print(f"SKIP REPAINT!!! {i=}, {j=}")
                            break

                        # os.makedirs("dump", exist_ok=True)
                        # torch.save(pseudo_x0, f"dump/pseudo_x0_ori_{i}_{j}.pt")

                        # alignment
                        if i < self._num_timesteps * 3 // 5:
                            M, pseudo_x0 = homography_estimation(
                                pseudo_x0,
                                warped_latents[batch_size:] if self.do_classifier_free_guidance else warped_latents,
                                warped_masks_sh < 0.5,
                                process_size=128,
                                lr=1e-2,
                                max_iters=100,
                                num_control_points=None,#num_frames//3,
                                fix_first_frame=True,
                                acceleration_penalty_weight=0.5,
                                padding_mode="border",
                            )
                            from kornia.geometry.transform.imgwarp import homography_warp
                            # latents = homography_warp(
                            #     latents.flatten(0, 1).float(),
                            #     M,
                            #     dsize=latents.shape[-2:],
                            #     padding_mode="reflection",
                            # ).reshape_as(latents).to(latents.dtype)
                            latents_ori = homography_warp(
                                latents_ori.flatten(0, 1).float(),
                                M,
                                dsize=latents.shape[-2:],
                                padding_mode="reflection",
                            ).reshape_as(latents_ori).to(latents_ori.dtype)

                        # os.makedirs("dump", exist_ok=True)
                        # torch.save(latents_ori, f"dump/latents_ori_{i}_{j}.pt")
                        # torch.save(pseudo_x0, f"dump/pseudo_x0_{i}_{j}.pt")

                        # resampling
                        if j < repaint_iter_num - 1:

                            sigma_t = self.scheduler.sigmas[i]
                            var_data = get_var_data(
                                attn_query,
                                attn_key,
                                warped_latents[batch_size:] if self.do_classifier_free_guidance else warped_latents,
                                warped_masks_sh,
                                kernel_radius=5,
                                use_first_frame=True,
                                channelwise=True,
                            )  # warped_latents[batch_size:, 0:1].var()

                            # os.makedirs("dump", exist_ok=True)
                            # torch.save(var_data, f"dump/var_data_{i}_{j}.pt")


                            var_data *= 3  # NOTE: MAGIC NUMBER!!!!!!!!!!!!


                            if i < self._num_timesteps // 2:
                                sigma_s = 0
                            else:
                                # deduce optimal sigma_s for RePaint
                                pseudo_x0 = pseudo_x0.float()
                                derivative = (latents_ori.float() - pseudo_x0) / sigma_t
                                identity = torch.ones_like(pseudo_x0[0, 0, :, 0, 0]).reshape(1, 1, -1, 1, 1)

                                k_spatial = 1
                                k_temporal = 1

                                var_pseudo_x0 = local_covariance_3D(pseudo_x0, pseudo_x0, k_spatial, k_temporal, channelwise=True)
                                var_derivative = local_covariance_3D(derivative, derivative, k_spatial, k_temporal, channelwise=True)
                                cov_pseudo_x0_derivative = local_covariance_3D(pseudo_x0, derivative, k_spatial, k_temporal, channelwise=True)

                                var_pseudo_x0 = guided_blur_2D(var_data, var_pseudo_x0)
                                var_derivative = guided_blur_2D(var_data, var_derivative)
                                cov_pseudo_x0_derivative = guided_blur_2D(var_data, cov_pseudo_x0_derivative)

                                coeff_A = var_derivative - identity
                                coeff_B = cov_pseudo_x0_derivative
                                coeff_C = var_pseudo_x0 - var_data

                                nunom = (-1) * coeff_B + torch.sign(coeff_B) * torch.sqrt(torch.relu(coeff_B.pow(2) - coeff_A * coeff_C))
                                sigma_s = safe_division_3D(nunom, coeff_A, k_spatial, k_temporal)

                                # endpoints check
                                sigma_s_left_eval = torch.abs(coeff_C)
                                sigma_s_right_eval = torch.abs(coeff_A * sigma_t**2 + 2 * coeff_B * sigma_t + coeff_C)
                                sigma_s_endpoints = torch.where(
                                    sigma_s_left_eval < sigma_s_right_eval,
                                    torch.zeros_like(sigma_s),
                                    torch.full_like(sigma_s, sigma_t),
                                )

                                sigma_s_endpoints_eval = torch.abs(coeff_A * sigma_s_endpoints**2 + 2 * coeff_B * sigma_s_endpoints + coeff_C)
                                sigma_s_eval = torch.abs(coeff_A * sigma_s**2 + 2 * coeff_B * sigma_s + coeff_C)
                                sigma_s = torch.where(sigma_s_eval < sigma_s_endpoints_eval, sigma_s, sigma_s_endpoints)

                                # postprocessing
                                sigma_s = guided_blur_2D(var_data, sigma_s)
                                sigma_s = torch.clamp(sigma_s, 0, sigma_t)


                                # torch.save(sigma_s, f"dump/sigma_s_{i}_{j}.pt")


                            # print(f"{sigma_s=}, {sigma_t=}")


                            # direct pasting
                            latents_mid = (sigma_s / sigma_t) * latents_ori + (1 - sigma_s / sigma_t) * pseudo_x0
                            warped_latents_noisy = warped_latents + sigma_s * randn_tensor(
                                warped_latents.shape, generator=generator, device=warped_latents.device, dtype=warped_latents.dtype)
                            latents_mid_pasted = warped_masks_sh * latents_mid + \
                                        (1 - warped_masks_sh) * (warped_latents_noisy[batch_size:] if self.do_classifier_free_guidance else warped_latents_noisy)

                            opt_std = 0.1
                            posterior_sigma = torch.inf #sigma_t**2 / opt_std
                            latents = self.stochastic_resample(
                                opt_zs=latents_mid_pasted,
                                ori_zt=latents_ori,
                                sigma_s=sigma_s,
                                sigma_t=sigma_t,
                                posterior_sigma=posterior_sigma,
                                generator=generator,
                            )

                        else:
                            pass


                        latents = latents.half()

                    del latents_ori

                    torch.cuda.empty_cache()


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)



                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # cpu offload
        import gc
        self.image_encoder.to("cpu")
        self.unet.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

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
            sigma_t: float | torch.Tensor,
            posterior_sigma: float | torch.Tensor = torch.inf,
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
            assert torch.all(0 <= sigma_s) and torch.all(sigma_s <= sigma_t), f"{sigma_s=}, {sigma_t=}"
        else:
            assert 0 <= sigma_s <= sigma_t, f"{sigma_s=}, {sigma_t=}"

        # cast everything to float32
        opt_zs = opt_zs.float()
        ori_zt = ori_zt.float() if ori_zt is not None else ori_zt
        sigma_s = sigma_s.float() if isinstance(sigma_s, torch.Tensor) else torch.tensor(sigma_s, dtype=torch.float32)
        sigma_t = sigma_t.float() if isinstance(sigma_t, torch.Tensor) else torch.tensor(sigma_t, dtype=torch.float32)
        posterior_sigma = posterior_sigma.float() if isinstance(posterior_sigma, torch.Tensor) else torch.tensor(posterior_sigma, dtype=torch.float32)

        noise = randn_tensor(opt_zs.shape, generator=generator, device=opt_zs.device, dtype=opt_zs.dtype)

        t_squared_minus_s_squared = (sigma_t ** 2 - sigma_s ** 2).relu()
        post_sigma_squared = posterior_sigma ** 2

        if posterior_sigma == torch.inf:
            return opt_zs + noise * t_squared_minus_s_squared**0.5
        else:
            assert ori_zt is not None

        denom = post_sigma_squared + t_squared_minus_s_squared
        mean = (post_sigma_squared / denom) * opt_zs + (t_squared_minus_s_squared / denom) * ori_zt
        std = posterior_sigma * (t_squared_minus_s_squared / (post_sigma_squared + t_squared_minus_s_squared)).relu().sqrt()
        return mean + noise * std


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
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu_memory_limit",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # limit GPU memory
    if args.gpu_memory_limit is not None:
        total_mem_gb = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
        fraction = args.gpu_memory_limit / total_mem_gb
        torch.cuda.set_per_process_memory_fraction(fraction, args.gpu)
        print(f"GPU memory upper limit was set to {args.gpu_memory_limit:.2f}GB ({fraction:.2%})")

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
    pipe.unet.record_layer_sublayer = [(2,1)]
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # load images
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    warped_masks = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}_mask.png")) for i in range(args.num_frames)]

    # inference
    # monitor = GPUMemoryMonitor(gpu_id=args.gpu)
    # monitor.start()

    svd_output = pipe(
        warped_images=warped_images,
        warped_masks=warped_masks,
        denoise_start_step=args.denoise_start_step,  # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        repaint_iter_num=args.repaint_iter_num,  # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ########
        num_frames=args.num_frames,
        decode_chunk_size=8,
        num_inference_steps=args.num_inference_steps,
        min_guidance_scale=args.min_guidance_scale,
        max_guidance_scale=args.max_guidance_scale,
        generator=torch.manual_seed(args.seed),
    )
    frames = svd_output.frames[0]

    # monitor.stop()
    # print(f"Peak GPU memory usage: {monitor.get_max_memory():.2f} GB")

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
