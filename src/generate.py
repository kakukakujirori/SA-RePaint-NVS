from typing import Callable, Dict, List, Optional, Union
import argparse
import glob
import inspect
import os
from dataclasses import dataclass

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils import BaseOutput, logging, load_image, export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline
from einops import rearrange
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from scheduling_euler_discrete import EulerDiscreteScheduler


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


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
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # print(image.shape,image.max())


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
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
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

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

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
        print(frames.shape)
        num_frames = num_frames
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

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
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
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
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        temp_cond,
        mask,
        # pose,
        lambda_ts,
        lr,
        weight_clamp,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)
        num_frames = 25

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0
        length = len(image)
        # 3. Encode input image
        with torch.no_grad():
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)



        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        mask = mask.cuda()
        mask = mask.unsqueeze(1).unsqueeze(0).repeat(1,1,4,1,1)
        temp_cond_list = []
        for i in range(len(temp_cond)):
            temp_cond_ = self.image_processor.preprocess(temp_cond[i], height=height, width=width)
            temp_cond_list.append(temp_cond_)
        temp_cond = torch.cat(temp_cond_list,dim=0)



        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise


        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            image_latents = self._encode_vae_image(image[0:1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance) # [2, 4, 72, 128]
            temp_cond_latents_list = []
            for i in range(temp_cond.shape[0]):
                temp_cond_latents_ = self._encode_vae_image(temp_cond[i:i+1,:,:,:], device, num_videos_per_prompt, do_classifier_free_guidance) # [12, 4, 72, 128]
                temp_cond_latents_ = rearrange(temp_cond_latents_, "(b f) c h w -> b f c h w",b=2)
                temp_cond_latents_list.append(temp_cond_latents_)
        temp_cond_latents = torch.cat(temp_cond_latents_list,dim=1)


        image_latents = rearrange(image_latents, "(b f) c h w -> b f c h w",f=1)

        temp_cond_latents  = torch.cat((image_latents,temp_cond_latents),dim=1)

        # .to(image_embeddings.dtype)
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents.repeat(1, num_frames, 1, 1, 1)




        factor_s = 5.6
        temp_cond_latents = temp_cond_latents/factor_s

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)


        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels



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



        # 7. Prepare guidance scale
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
                grads = []
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input_test1=latent_model_input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t,step_i=i)
                latent_model_input_test2=latent_model_input
                latent_model_input = torch.cat([latent_model_input[0:1], image_latents[1:2]], dim=2)
                latent_model_input2 = latent_model_input
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
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,:72]
                            mask1 = mask[0:1,:,:,:40,:72]
                        elif ii ==1:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,:72]
                            latents1 = latents[0:1,:,:,32:,:72]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,:72]
                            mask1 = mask[0:1,:,:,32:,:72]
                        elif ii ==2:
                            latent_model_input1 = latent_model_input[0:1,:,:,:40,56:]
                            latents1 = latents[0:1,:,:,:40,56:]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,56:]
                            mask1 = mask[0:1,:,:,:40,56:]
                        elif ii ==3:
                            latent_model_input1 = latent_model_input[0:1,:,:,32:,56:]
                            latents1 = latents[0:1,:,:,32:,56:]
                            temp_cond_latents1 = temp_cond_latents[:2,:,:,32:,56:]
                            mask1 = mask[0:1,:,:,32:,56:]
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

                        output = self.scheduler.step_single(noise_pred_t, t, latents1,temp_cond_latents1,mask1,lambda_ts,step_i=i,lr=lr,weight_clamp=weight_clamp,compute_grad=True)
                        grad = output.grad
                        grads.append(grad)

                grads1 = torch.cat((grads[0],grads[1][:,:,:,8:,:]),-2)
                grads2 = torch.cat((grads[2],grads[3][:,:,:,8:,:]),-2)
                grads3 = torch.cat((grads1,grads2[:,:,:,:,16:]),-1)
                latents = latents - grads3.half()





                with torch.no_grad():


                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t,step_i=i)
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)


                    latents = self.scheduler.step_single(noise_pred, t, latents,temp_cond_latents,mask,lambda_ts,step_i=i,compute_grad=False).prev_sample


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
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()


        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


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
        "--lr",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--weight_clamp",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    # load images
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    image_o, cond_images = warped_images[0], warped_images[1:]

    # load masks
    masks = []
    for i in range(1, args.num_frames):
        mask_erosion = PIL.Image.open(os.path.join(args.trajectory_folder, f'{i:04d}_mask.png'))
        mask_erosion_ = np.array(mask_erosion)/255.
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1

        mask_erosion = np.mean(mask_erosion_,axis = -1)
        mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
        mask_erosion = np.mean(mask_erosion,axis=2)
        mask_erosion[mask_erosion < 0.2] = 0
        mask_erosion[mask_erosion >= 0.2] = 1
        masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))
    masks = torch.concat(masks)

    # TODO: load camera poses
    # TODO: load depths

    # calculate lambda
    sigma_list = sigma_list = np.load(f'sigmas/sigmas_{args.num_inference_steps}.npy').tolist()
    lambda_ts = search_hypers(sigma_list,args.num_frames)
    lambda_ts = torch.tensor(lambda_ts)

    pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    frames = pipe(
        [image_o],
        temp_cond=cond_images,
        mask=masks,
        # poses=poses,
        # intrinsics=intrinsics,
        # depths=depths,
        lambda_ts=lambda_ts,
        lr=args.lr,
        weight_clamp=args.weight_clamp,
        num_frames=args.num_frames,
        decode_chunk_size=8,
        num_inference_steps=args.num_inference_steps,
        generator=torch.manual_seed(args.seed),
    ).frames[0]

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
