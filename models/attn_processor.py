import PIL
import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import PIL.Image
import torch
from torch import nn
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, ControlNetModel, ImageProjection, MultiAdapter, T2IAdapter
from diffusers.models.controlnet import ControlNetOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    PIL_INTERPOLATION, 
    logging, 
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, StableDiffusionMixin, ImagePipelineOutput
import time
import sys
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid
import os

from torch import optim
import copy
import torch.nn.functional as F
import json
import kornia

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder

from functools import wraps
import logging

import types
import math

class PILOT_CrossAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        attn_mask = None,
        mask_ca = False,
        attn_loss_bank = [],
        text_input_ids = [],
        *args,
        **cross_attention_kwargs,
    ) -> torch.FloatTensor:
        # print("shape of hidden_states: ",hidden_states.shape)
        if len(args) > 0 or cross_attention_kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        embedding_shape = encoder_hidden_states.shape[-2]
        latent_shape = hidden_states.shape[-2]
        
        if (mask_ca) and (attn_mask is not None):
            attention_mask=rearrange(attn_mask[str(hidden_states.shape[-2])],"h w -> () () (h w) ()")
            attention_mask = torch.cat([torch.ones((1, 1, latent_shape, 1)).to('cuda'),  # 0th token: <bos>, leave it as-is
                                attention_mask.repeat(1,1,1,embedding_shape-1)], dim=-1) # tokens only attend to fg area
            attention_mask = attention_mask.squeeze(dim=0)
            attention_mask = attention_mask.repeat(1,1,1)
        else:
            attention_mask = torch.ones((1, latent_shape, embedding_shape)).to('cuda')
        attention_mask = attention_mask.to(hidden_states.dtype)


        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        ############################### source attention processor ######################
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_probs = attn.get_attention_scores(attn.head_to_batch_dim(query), attn.head_to_batch_dim(key))

        if (attention_mask is None) or torch.all(attention_mask[:,:,1:] == 0) or torch.all(attention_mask[:,:,1:] == 1):
            attn_loss_bank.append(0)
        else:
            token_indices = [i for i, value in enumerate(text_input_ids[0]) if value != 49406 and value != 49407]
            reverse_attention_probs = attention_probs * (1-attention_mask)
            reverse_score = torch.sum(reverse_attention_probs[:,:,token_indices]) / (torch.sum(1-attention_mask[:,:,1]) * attention_probs.shape[0])

            in_attention_probs = attention_probs * attention_mask
            in_score = torch.sum(in_attention_probs[:,:,token_indices]) / (torch.sum(attention_mask[:,:,1]) * attention_probs.shape[0])
            attn_loss_bank.append(reverse_score - in_score)
            
        # whether to add attention mask to the attention score
        attention_probs = attention_probs * attention_mask

        # print("shape of attention score: ",attention_probs.shape)
        hidden_states = torch.bmm(attention_probs, attn.head_to_batch_dim(value))
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class PILOT_SelfAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        attn_mask = None,
        mask_ca = False,
        attn_loss_bank = [],
        text_input_ids = [],
        *args,
        **cross_attention_kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or cross_attention_kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = None

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        # if attention_mask!=None:
        #     if (torch.any(attention_mask)==False):
        #         attention_mask=torch.ones((1, 1, query.shape[-2],  key.shape[-2]), dtype=torch.bool).to('cuda')

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # print("shape of query: ",query.shape)
        # print("shape of key: ",key.shape)
        # print("shape of value: ",value.shape)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class PILOT_IPAdapterAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        scale = 1.0,
        ip_adapter_masks = None,
        attn_mask = None,
        mask_ca = False,
        attn_loss_bank = [],
        text_input_ids = [],
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        embedding_shape = encoder_hidden_states.shape[-2]
        latent_shape = hidden_states.shape[-2]

        if (mask_ca) and (attn_mask is not None):
            attention_mask=rearrange(attn_mask[str(hidden_states.shape[-2])],"h w -> () () (h w) ()")
            attention_mask = torch.cat([torch.ones((1, 1, latent_shape, 1)).to('cuda'),  # 0th token: <bos>, leave it as-is
                                attention_mask.repeat(1,1,1,embedding_shape-1)], dim=-1) # tokens only attend to fg area
            attention_mask = attention_mask.squeeze(dim=0)
            attention_mask = attention_mask.repeat(1,1,1)
        else:
            attention_mask = torch.ones((1, latent_shape, embedding_shape)).to('cuda')
        attention_mask = attention_mask.to(hidden_states.dtype)

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        attention_probs = attn.get_attention_scores(attn.head_to_batch_dim(query), attn.head_to_batch_dim(key))

        if (attention_mask is None) or torch.all(attention_mask[:,:,1:] == 0) or torch.all(attention_mask[:,:,1:] == 1):
            attn_loss_bank.append(0)
        else:
            token_indices = [i for i, value in enumerate(text_input_ids[0]) if value != 49406 and value != 49407]
            reverse_attention_probs = attention_probs * (1-attention_mask)
            reverse_score = torch.sum(reverse_attention_probs[:,:,token_indices]) / (torch.sum(1-attention_mask[:,:,1]) * attention_probs.shape[0])

            in_attention_probs = attention_probs * attention_mask
            in_score = torch.sum(in_attention_probs[:,:,token_indices]) / (torch.sum(attention_mask[:,:,1]) * attention_probs.shape[0])
            attn_loss_bank.append(reverse_score - in_score)
            
        # whether to add attention mask to the attention score
        attention_probs = attention_probs * attention_mask

        # print("shape of attention score: ",attention_probs.shape)
        hidden_states = torch.bmm(attention_probs, attn.head_to_batch_dim(value))
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
    
        return hidden_states


def revise_pilot_unet_cross_attention_forward(unet):
    def change_forward(unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                if "IPAdapter" in layer.processor.__class__.__name__:
                    ip_adapter=PILOT_IPAdapterAttnProcessor(layer.processor.hidden_size,
                                                                    layer.processor.cross_attention_dim,
                                                                    layer.processor.num_tokens,
                                                                    layer.processor.scale,
                                                                     )
                    ip_adapter.to_k_ip = layer.processor.to_k_ip
                    ip_adapter.to_v_ip = layer.processor.to_v_ip
                    layer.set_processor(ip_adapter)
                else:
                    layer.set_processor(PILOT_CrossAttnProcessor())
            else:
                change_forward(layer)

    # use this to ensure the order
    change_forward(unet.down_blocks)
    change_forward(unet.mid_block)
    change_forward(unet.up_blocks)

def revise_pilot_unet_self_attention_forward(unet):
    def change_forward(unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn1' in name:
                layer.set_processor(PILOT_SelfAttnProcessor())
            else:
                change_forward(layer)

    # use this to ensure the order
    change_forward(unet.down_blocks)
    change_forward(unet.mid_block)
    change_forward(unet.up_blocks)


def revise_pilot_unet_attention_forward(unet):
    revise_pilot_unet_cross_attention_forward(unet)
    revise_pilot_unet_self_attention_forward(unet)
