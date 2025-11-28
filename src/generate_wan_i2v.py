# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, html, os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import lovely_tensors
import numpy as np
import PIL
import regex as re
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import BaseOutput, logging, export_to_video, is_ftfy_available, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from covariance import guided_blur_2D, local_covariance_2D, local_covariance_3D, safe_division_3D
from gpu_memory_monitor import GPUMemoryMonitor
from scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformer_wan import MyWanTransformer3DModel
from warp import homography_estimation


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

lovely_tensors.monkey_patch()

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


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
    batch, num_frames, channel, height, width = warped_latents.shape
    scale_factor = int(round((height * width // q.shape[-2])**0.5))
    assert batch == 1
    assert height * width == q.shape[-2] * scale_factor**2, f"{height=}, {width=}, {q.shape=}, {scale_factor=}"
    assert num_frames == q.shape[0] == k.shape[0], f"{warped_latents.shape=}, {q.shape=}, {k.shape=}"
    h, w = height // scale_factor, width // scale_factor

    # local spatial variance of warped_masks_sh (NOTE: select channelwise=False on purpose; see the NOTE below)
    warped_variance = local_covariance_2D(warped_latents, warped_latents, k=kernel_radius, channelwise=False)
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
    # NOTE: If the channel size is too small (e.g., 4), scaled_dot_product_attention defaults to the MATH backend, which has very high memory cost.
    #       To avoid this, we force the [EFFICIENT/FLASH]_ATTENTION backend, which should be auto-selected when channelwise=False (resulting in 16 channels).
    #       However, if channelwise=True and the channel size is 4, it raises "No available kernel"; this can work as an assertion check.
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
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

    # extract diagonal components
    if channelwise:
        var_data = rearrange(var_data, "b f (c1 c2) h w -> b f c1 c2 h w", c1=channel, c2=channel)
        var_data = torch.diagonal(var_data, dim1=2, dim2=3).permute(0, 1, 4, 2, 3) # [b, f, c, h, w]

    return var_data


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for Wan pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. In two-stage denoising,
            `transformer` handles high-noise stages and `transformer_2` handles low-noise stages. If not provided, only
            `transformer` is used.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        # self.register_to_config()

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        warped_images,
        warped_masks,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if warped_images is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `warped_images`: {warped_images} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if warped_images is None and image_embeds is None:
            raise ValueError(
                "Provide either `warped_images` or `prompt_embeds`. Cannot leave both `warped_images` and `image_embeds` undefined."
            )
        if warped_images is not None and not isinstance(warped_images, torch.Tensor) and not isinstance(warped_images[0], PIL.Image.Image):
            raise ValueError(f"`warped_images` has to be of type `torch.Tensor` or `list[PIL.Image.Image]` but is {type(warped_images)}")
        if warped_masks is not None and not isinstance(warped_masks, torch.Tensor) and not isinstance(warped_masks[0], PIL.Image.Image):
            raise ValueError(f"`warped_masks` has to be of type `torch.Tensor` or `list[PIL.Image.Image]` but is {type(warped_masks)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        warped_images: PipelineImageInput,
        warped_masks: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        denoise_start_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # prepare conditioning
        warped_images = rearrange(warped_images, "f c h w -> () c f h w", c=3)  # [batch_size, channels, frames, height, width]
        video_condition = warped_images.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # prepare masks
        warped_masks = rearrange(warped_masks, "f c h w -> () c f h w", c=3)
        warped_masks = warped_masks.mean(dim=1, keepdim=True)
        warped_masks_sh = rearrange(warped_masks, "() () f (nh ph) (nw pw) -> () () f nh nw (ph pw)", ph=16, pw=16)
        warped_masks_sh = warped_masks_sh.mean(dim=-1)
        assert warped_masks_sh.shape == (1, 1, num_frames, latent_height, latent_width)

        first_mask = warped_masks_sh[:, :, 0:1]
        condition_mask = rearrange(warped_masks_sh[:, :, 1:], "() () (f pf) nh nw -> () () f pf nh nw", pf=4)
        condition_mask = condition_mask.mean(dim=3)
        condition_mask = torch.cat([first_mask, condition_mask], dim=2)
        condition_mask = torch.clip(condition_mask * 5, 0, 1)
        assert condition_mask.shape == (1, 1, num_latent_frames, latent_height, latent_width)

        # prepare latents
        if denoise_start_step is not None:
            if latents is not None:
                print("Warning: when denoise_start_step is set, 'latents' argument is ignored.")
            sigma_t = self.scheduler.sigmas[denoise_start_step]  # NOTE: sigmas are accessible after scheduler.set_timesteps is called
            latents = (1 - sigma_t) * latent_condition + sigma_t * randn_tensor(
                latent_condition.shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

        return latents, latent_condition, condition_mask


    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        warped_images: List[PIL.Image.Image],
        warped_masks: List[PIL.Image.Image],
        denoise_start_step: Optional[int],
        repaint_iter_num: int,
        ########
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            warped_images (`List[PIL.Image.Image]`):
                The input images to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            warped_masks (`List[PIL.Image.Image]`):
                The input masks corresponding to the warped images. Must be a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            warped_images,
            warped_masks,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        if denoise_start_step is not None:
            assert denoise_start_step < num_inference_steps

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        device = self._execution_device

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
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        warped_images = self.video_processor.preprocess(warped_images, height=height, width=width).to(device, dtype=torch.float32)
        warped_masks = self.video_processor.preprocess(warped_masks, height=height, width=width).to(device, dtype=torch.float32)
        warped_masks = (warped_masks > 0).float() # [-1, 1] -> {0, 1}

        latents_outputs = self.prepare_latents(
            warped_images,
            warped_masks,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            denoise_start_step
        )
        latents, condition, condition_mask = latents_outputs
        first_frame_mask = torch.ones_like(condition_mask)
        first_frame_mask[:, :, 0:1] = 0

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if (denoise_start_step is not None) and i < denoise_start_step:
                    progress_bar.update()
                    continue

                for j in range(repaint_iter_num):
                    latents_ori = rearrange(latents, "b c f h w -> b f c h w").float()

                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

                    # Attention weighting
                    base_weight = min(1, 0.0 + i * 8 / self._num_timesteps)  # TODO: MAGIC NUMBER!!!
                    weight_map = (1 - condition_mask) * (1 - base_weight) + base_weight

                    # SEG
                    # query_blur_sigma_invalid_max = 2  # TODO: MAGIC NUMBER!!!
                    # query_blur_sigma_invalid_min = 2  # TODO: MAGIC NUMBER!!!
                    # query_blur_sigma_invalid = query_blur_sigma_invalid_min + (query_blur_sigma_invalid_max - query_blur_sigma_invalid_min) * i / self._num_timesteps
                    # query_blur_sigma_valid = 4  # TODO: MAGIC NUMBER!!!
                    # query_blur_sigma = (1 - condition_mask) * query_blur_sigma_valid + condition_mask * query_blur_sigma_invalid

                    with self.transformer.cache_context("cond"):
                        self.transformer.latent_shape_ = latents.shape[-3:]
                        self.transformer.inject(weight_map)
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            record_attention=True,
                        )[0]

                        # retrieve qk
                        attn_query = self.transformer.record_query_[0]
                        attn_key = self.transformer.record_key_[0]
                        attn_value = self.transformer.record_value_[0]
                        os.makedirs("dump", exist_ok=True)
                        torch.save(attn_query, f"dump/query_{i}_{j}_{0}.pt")
                        torch.save(attn_key, f"dump/key_{i}_{j}_{0}.pt")
                        torch.save(attn_value, f"dump/value_{i}_{j}_{0}.pt")

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        use_seg = False #i % 3 == 0

                        if use_seg:
                            with self.transformer.cache_context("uncond"):
                                self.transformer.latent_shape_ = latents.shape[-3:]
                                self.transformer.inject(weight_map, query_blur_sigma=query_blur_sigma)
                                noise_uncond = self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                    record_attention=False,
                                )[0]
                        else:
                            with self.transformer.cache_context("uncond"):
                                self.transformer.latent_shape_ = latents.shape[-3:]
                                self.transformer.inject(weight_map)
                                noise_uncond = self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                    record_attention=False,
                                )[0]

                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    out = self.scheduler.step(noise_pred, t, latents, step_i=i)
                    pseudo_x0 = rearrange(out.pred_original_sample, "b c f h w -> b f c h w")
                    latents = out.prev_sample

                    if i >= self._num_timesteps * 4//5:
                        # This avoids flickering around small masks
                        print(f"SKIP SA-REPAINT!!! {i=}, {j=}")
                        break

                    os.makedirs("dump", exist_ok=True)
                    torch.save(pseudo_x0, f"dump/pseudo_x0_ori_{i}_{j}.pt")

                    # alignment
                    if i < self._num_timesteps * 1 // 5:
                        M, pseudo_x0 = homography_estimation(
                            pseudo_x0,
                            rearrange(condition, "b c f h w -> b f c h w"),
                            rearrange(condition_mask, "b c f h w -> b f c h w") < 0.5,
                            process_size=128,
                            lr=1e-2,
                            max_iters=100,
                            num_control_points=None,#num_frames//3,
                            fix_first_frame=True,
                            acceleration_penalty_weight=0.5,
                            padding_mode="border",
                        )
                        from kornia.geometry.transform.imgwarp import homography_warp
                        latents_ori = homography_warp(
                            rearrange(latents_ori, "b f c h w -> (b f) c h w"),
                            M,
                            dsize=latents.shape[-2:],
                            padding_mode="border",
                        )
                        latents_ori = rearrange(latents_ori, "(b f) c h w -> b f c h w", b=latents.shape[0])

                        # os.makedirs("dump", exist_ok=True)
                        # torch.save(M, f"dump/homography_{i}_{j}.pt")
                        # torch.save(pseudo_x0, f"dump/pseudo_x0_{i}_{j}.pt")
                        # torch.save(latents_ori, f"dump/latents_ori_{i}_{j}.pt")

                    # resampling
                    if j < repaint_iter_num - 1:
                        sigma_t = self.scheduler.sigmas[i]

                        if i < self._num_timesteps * 0 // 2:
                            sigma_s = torch.tensor([0], device=sigma_t.device, dtype=sigma_t.dtype).reshape(1, 1, 1, 1, 1)
                        else:
                            # approximate Var[z0] by attention qk-similarity
                            var_data = get_var_data(
                                attn_query.float(),
                                attn_key.float(),
                                rearrange(condition, "b c f h w -> b f c h w"),
                                rearrange(condition_mask, "b c f h w -> b f c h w"),
                                kernel_radius=5,
                                use_first_frame=True,
                                channelwise=True,
                            )

                            # os.makedirs("dump", exist_ok=True)
                            # torch.save(var_data, f"dump/var_data_{i}_{j}.pt")

                            var_data *= 1.5  # NOTE: MAGIC NUMBER!!!!!!!!!!!!

                            # deduce optimal sigma_s for RePaint
                            pseudo_x0 = pseudo_x0.float()
                            derivative = (latents_ori - pseudo_x0) / sigma_t
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

                            # check two roots
                            nunom_pos = (-1) * coeff_B + torch.sign(coeff_B) * torch.sqrt(torch.relu(coeff_B.pow(2) - coeff_A * coeff_C))
                            sigma_s_pos = safe_division_3D(nunom_pos, coeff_A, k_spatial, k_temporal)
                            sigma_s_pos = torch.clamp(sigma_s_pos, 0, sigma_t)

                            nunom_neg = (-1) * coeff_B - torch.sign(coeff_B) * torch.sqrt(torch.relu(coeff_B.pow(2) - coeff_A * coeff_C))
                            sigma_s_neg = safe_division_3D(nunom_neg, coeff_A, k_spatial, k_temporal)
                            sigma_s_neg = torch.clamp(sigma_s_neg, 0, sigma_t)

                            sigma_s_pos_eval = torch.abs(coeff_A * sigma_s_pos**2 + 2 * coeff_B * sigma_s_pos + coeff_C)
                            sigma_s_neg_eval = torch.abs(coeff_A * sigma_s_neg**2 + 2 * coeff_B * sigma_s_neg + coeff_C)
                            sigma_s = torch.where(
                                sigma_s_pos_eval <= sigma_s_neg_eval + 1e-6, # NOTE: if two solutions are available, adopt sigma_s_pos
                                sigma_s_pos,
                                sigma_s_neg,
                            )

                            # endpoints check
                            sigma_s_left_eval = torch.abs(coeff_C)
                            sigma_s_right_eval = torch.abs(coeff_A * sigma_t**2 + 2 * coeff_B * sigma_t + coeff_C)
                            sigma_s_endpoints = torch.where(
                                sigma_s_left_eval <= sigma_s_right_eval,
                                torch.zeros_like(sigma_s),
                                torch.full_like(sigma_s, sigma_t),
                            )

                            sigma_s_endpoints_eval = torch.abs(coeff_A * sigma_s_endpoints**2 + 2 * coeff_B * sigma_s_endpoints + coeff_C)
                            sigma_s_eval = torch.abs(coeff_A * sigma_s**2 + 2 * coeff_B * sigma_s + coeff_C)
                            sigma_s = torch.where(sigma_s_eval < sigma_s_endpoints_eval, sigma_s, sigma_s_endpoints)

                            # postprocessing
                            sigma_s = guided_blur_2D(var_data, sigma_s)
                            sigma_s = torch.clamp(sigma_s, 0, sigma_t)

                        print(f"{sigma_s=}, {sigma_t=}")

                        # direct pasting
                        pseudo_x0 = rearrange(pseudo_x0, "b f c h w -> b c f h w")
                        sigma_s = rearrange(sigma_s, "b f c h w -> b c f h w")
                        latents_mid = pseudo_x0 + sigma_s * noise_pred  # NOTE: `noise_pred = eps - x0` for flow_prediction
                        condition_noisy = (1 - sigma_s) * condition + sigma_s * randn_tensor(
                            condition.shape, generator=generator, device=condition.device, dtype=condition.dtype)
                        latents_mid_pasted = condition_mask * latents_mid + (1 - condition_mask) * condition_noisy

                        coeff_data = (1 - sigma_t) / (1 - sigma_s + 1e-6)
                        coeff_noise = (sigma_t**2 - (coeff_data ** 2) * sigma_s**2).sqrt()
                        latents = coeff_data * latents_mid_pasted + coeff_noise * randn_tensor(
                            latents_mid_pasted.shape, generator=generator, device=latents_mid_pasted.device, dtype=latents_mid_pasted.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = (1 - condition_mask) * condition + condition_mask * latents

        # cpu offload
        import gc
        self.transformer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


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
        "--guidance_scale",
        type=float,
        default=5.0,
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
        default=2,
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
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = MyWanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # load captioner
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    )

    # load images
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    warped_masks = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}_mask.png")) for i in range(args.num_frames)]

    # inference
    # monitor = GPUMemoryMonitor(gpu_id=args.gpu)
    # monitor.start()

    # qk extraction setting
    pipe.transformer.record_blocks = [15]
    assert max(pipe.transformer.record_blocks) < len(pipe.transformer.blocks), f"Must be lower than {len(pipe.transformer.blocks)=}"

    # get caption
    captioner.to(device)
    captioner_inputs = caption_processor(images=warped_images[0], return_tensors="pt").to(device, torch.float16)
    generated_ids = captioner.generate(**captioner_inputs)
    generated_text = caption_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    prompt = generated_text + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    captioner.cpu()
    print(prompt)

    frames = pipe(
        warped_images=warped_images,
        warped_masks=warped_masks,
        denoise_start_step=args.denoise_start_step,
        repaint_iter_num=args.repaint_iter_num,
        prompt=prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        height=704,
        width=1280,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.manual_seed(args.seed),
    ).frames[0]

    # monitor.stop()
    # print(f"Peak GPU memory usage: {monitor.get_max_memory():.2f} GB")

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
