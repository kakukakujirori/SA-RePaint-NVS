import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput
from einops import rearrange
from kornia.filters import gaussian_blur2d

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MyUNet(UNetSpatioTemporalConditionModel):
    """
    Modified from SVD implementation
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_spatio_temporal_condition.py
    """
    def inject(self, weight: Optional[torch.Tensor] = None, query_blur_sigma: float = 0.0):
        # for (layer, downsample_block) in enumerate(self.down_blocks):  # ['CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal']
        #     if layer == 3:
        #         continue
        #     for (sublayer, trans) in enumerate(downsample_block.attentions):  # ['TransformerSpatioTemporalModel', 'TransformerSpatioTemporalModel']
        #         basictrans = trans.transformer_blocks[0]  # BasicTransformerBlock (spatial)
        #         basictrans.attn1.processor = self.my_self_attention(3 - layer, sublayer, mask)

        # for (sublayer, trans) in enumerate(self.mid_block.attentions):  # ['TransformerSpatioTemporalModel', 'TransformerSpatioTemporalModel']
        #     basictrans = trans.transformer_blocks[0]  # BasicTransformerBlock (spatial)
        #     basictrans.attn1.processor = self.my_self_attention(0, sublayer, mask)

        for (layer, upsample_block) in enumerate(self.up_blocks):  # ['UpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal']
            if layer == 0:
                continue
            for (sublayer, trans) in enumerate(upsample_block.attentions):  # ['TransformerSpatioTemporalModel', 'TransformerSpatioTemporalModel', 'TransformerSpatioTemporalModel']
                basictrans = trans.transformer_blocks[0]  # BasicTransformerBlock (spatial)
                basictrans.attn1.processor = self.my_self_attention(layer, sublayer, weight, query_blur_sigma)
                # tmpbasictrans = trans.temporal_transformer_blocks[0]  # TemporalBasicTransformerBlock (temporal)
                # tmpbasictrans.attn1.processor = self.my_temporal_attention(layer, sublayer, mask)

    record_layer_sublayer: list[tuple[int, int]] = []

    latent_shape_: Optional[list[int]] = None
    record_attention_: Optional[bool] = None
    record_query_ = []
    record_key_ = []
    record_value_ = []

    def my_self_attention(self, layer: int, sublayer: int, weight: Optional[torch.Tensor] = None, query_blur_sigma: float = 0.0) -> AttentionProcessor:
        compress_factor = [8, 4, 2, 1][layer]
        h = self.latent_shape_[-2] // compress_factor
        w = self.latent_shape_[-1] // compress_factor

        if (weight is not None) and isinstance(weight, torch.Tensor):
            weight = rearrange(weight, "batch frames () h w -> (batch frames) () h w", h=self.latent_shape_[-2], w=self.latent_shape_[-1])
            weight = F.interpolate(weight.float(), size=(h, w), mode="bilinear")
            weight_reshaped = rearrange(weight, "bf () h w -> bf () (h w) ()")

        def processor(
            attn,
            hidden_states,  # (batch_size=num_frames(x2), height*width, channels)
            encoder_hidden_states = None,
            attention_mask = None,
            temb = None,
        ):
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if self.record_attention_ and (layer, sublayer) in self.record_layer_sublayer:
                self.record_query_.append(query)
                self.record_key_.append(key)
                # self.record_value_.append(value)

            if weight is not None:
                key *= weight_reshaped.repeat(key.shape[0] // weight_reshaped.shape[0], 1, 1, 1)

            if query_blur_sigma > 0.0:
                kernel_size = math.ceil(6 * query_blur_sigma) + 1 - math.ceil(6 * query_blur_sigma) % 2
                query_reshaped = rearrange(query, "bf heads (h w) c -> bf (heads c) h w", h=h, w=w)
                query_blurred = torch.where(
                    weight > 0.999,
                    gaussian_blur2d(query_reshaped, kernel_size=(kernel_size, kernel_size), sigma=(query_blur_sigma, query_blur_sigma)),
                    gaussian_blur2d(query_reshaped, kernel_size=(3, 3), sigma=(0.5, 0.5))
                )
                query = rearrange(query_blurred, "bf (heads c) h w -> bf heads (h w) c", c=query.shape[-1])
                query = query.contiguous()
                del query_reshaped, query_blurred

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states
        return processor

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        record_attention: bool = False,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        if self.latent_shape_ is None:
            self.latent_shape_ = sample.shape
        else:
            assert self.latent_shape_ == sample.shape, f"Expected sample shape {self.latent_shape_}, but got {sample.shape}"

        self.record_attention_ = record_attention
        if self.record_attention_:
            self.record_query_ = []
            self.record_key_ = []
            self.record_value_ = []

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
