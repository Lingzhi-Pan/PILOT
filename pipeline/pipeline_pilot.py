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
import torch.nn.functional as F
import kornia

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from models.attn_processor import revise_pilot_unet_attention_forward

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    # dilation and mask
    mask = kornia.morphology.dilation(mask, kernel=torch.ones((17,17)))
    # masked_image = (image+1)/2 * (mask > 0.5)
    # no mask
    masked_image = (image+1)/2
    masked_image = masked_image*2 - 1

    return mask, masked_image


def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image

class PilotPipeline(DiffusionPipeline,
                    StableDiffusionMixin,
                    TextualInversionLoaderMixin,
                    LoraLoaderMixin,
                    IPAdapterMixin,
                    FromSingleFileMixin,):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]]=None,
        adapter: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]]=None,
        image_encoder: CLIPVisionModelWithProjection = None,
        # requires_safety_checker: bool = True,
    ):
        super().__init__()
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
            
        if isinstance(adapter, (list, tuple)):
            adapter = MultiAdapter(adapter)
        
        revise_pilot_unet_attention_forward(unet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            adapter=adapter,
            controlnet=controlnet,
            scheduler=scheduler,
            # safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # self.register_to_config(requires_safety_checker=requires_safety_checker)
        # print("unet: ",self.unet)
        
        self.t2i_scale = 0.0

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()
        
    def set_t2i_adapter_scale(self, t2i_scale):
        if isinstance(t2i_scale, list):
            self.t2i_scale = t2i_scale[0]
        if isinstance(t2i_scale, float):
            self.t2i_scale = t2i_scale

    def set_controlnet_scale(self, controlnet_scale):
        if isinstance(controlnet_scale, list):
            self.controlnet_scale = controlnet_scale[0]
        if isinstance(controlnet_scale, float):
            self.controlnet_scale = controlnet_scale


    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        # if self.safety_checker is not None:
        #     cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # if self.safety_checker is not None:
        #     _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        text_embedding_only: bool = False,
        null_embedding_only: bool = False,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding = "max_length",
                max_length = max_length,
                truncation = True,
                return_tensors = "pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask = attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            if null_embedding_only:
                return negative_prompt_embeds
        
        if do_classifier_free_guidance and (text_embedding_only == False):
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype = self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)
        return prompt_embeds

    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    # def run_safety_checker(self, image, device, dtype):
    #     if self.safety_checker is not None:
    #         safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
    #         image, has_nsfw_concept = self.safety_checker(
    #             images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
    #         )
    #     else:
    #         has_nsfw_concept = None
    #     return image, has_nsfw_concept

    def decode_latents(self, latents, return_type="numpy"):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        if return_type == "tensor":
            return image
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def encode_image(self, img, generator,return_type="tensor"):
        posterior = self.vae.encode(img).latent_dist
        latents = posterior.sample(generator=generator)
        latents = self.vae.config.scaling_factor * latents
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
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

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            masked_image_latents = [
                self.vae.encode(masked_image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def encode_ipadapter_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds


    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_ipadapter_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def tensor_to_numpy(self,x):
        x = (x/2+0.5).clamp(0,1)
        x = x.cpu().permute(0,2,3,1).float().numpy()
        return x

    def lr_scheduler(self, lr_f, lr=0.2):
        lr_xt = []
        if lr_f == "linear":
            coef0 = 10
            coefT = 0.01
            lr_xt = [(coefT-coef0)/1000*i+coef0 for i in range(1000)]
        elif lr_f == "exp":
            lr_xt = [lr*1.002**(1000-i) for i in range(1000)]
        elif lr_f == "constant":
            lr_xt = 1000*[lr]
        else:
            lr_xt = None
        return lr_xt


    def coef_scheduler(self, coef_f, coef_start=0.2):
        coef_xt = []
        if coef_f == "constant":
            coef_xt = 1000*[coef_start]
        elif coef_f == "linear":
            coef_xt = 300*[0.1*coef_start] + list(np.linspace(0.1*coef_start,coef_start,700))
        else:
            coef_xt = None
        return coef_xt
    

    def cal_bg_loss(self, pred_image, image, mask, sum_all=True):
        pred_image = (pred_image+1)*0.5
        image = (image + 1) * 0.5

        if sum_all:
            loss = torch.sum((image * (mask>0.5) - pred_image * (mask>0.5)) ** 2)
        else:
            loss = torch.sum((image * (mask>0.5) - pred_image * (mask>0.5)) ** 2, dim = (1,2,3))
        return loss

    def optimize_xt(self, 
                    x, 
                    image, 
                    mask, 
                    t, 
                    cfg,
                    do_classifier_free_guidance = True,
                    lr_f = "exp",
                    momentum = 0.7,
                    lr = 0.2,
                    no_op = False,
                    coef = 0.05,
                    coef_f = "constant",
                    attention_mask = None,
                    prompt_embeds = None,
                    cond_image = None,
                    lr_warmup = 0.01,
                    num_gradient_ops = 10,
                    down_intrablock_additional_residuals = None,
                    adapter_state = None,
                    x_m = torch.tensor([0]),
                    added_cond_kwargs = None,
                    model_list = []
                    ):
        x_m = x_m.to('cuda')
        do_classifier_free_guidance = cfg >= 1.0
        if no_op == False:
            with torch.enable_grad():
                lr_xt = self.lr_scheduler(lr_f = lr_f,lr=lr)
                lr_xt[self.scheduler.timesteps[0]] = lr_warmup
                coef = self.coef_scheduler(coef_f=coef_f, coef_start=coef)
                x = x.requires_grad_()
                
                if num_gradient_ops != 0:
                    for step in range(num_gradient_ops):
                        x_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                        down_block_res_samples = None
                        mid_block_res_sample = None
                        if "controlnet" in model_list:
                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                x_input,
                                t,
                                encoder_hidden_states = prompt_embeds,
                                controlnet_cond = cond_image,
                                conditioning_scale = self.controlnet_scale,
                                return_dict = False,
                            )
                        if "t2iadapter" in model_list:
                            down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                        if do_classifier_free_guidance:
                            noise_pred = self.unet(x_input,
                                                t, 
                                                encoder_hidden_states = prompt_embeds, 
                                                down_block_additional_residuals = down_block_res_samples,
                                                mid_block_additional_residual = mid_block_res_sample,
                                                down_intrablock_additional_residuals = down_intrablock_additional_residuals, # Added for T2I adapter
                                                added_cond_kwargs = added_cond_kwargs,
                                                cross_attention_kwargs = {
                                                    'attn_mask': attention_mask,
                                                    'mask_ca': True,
                                                    'mask_sa': False
                                                },
                                                ).sample
                        else:
                            noise_pred = self.unet(x_input,
                                                t, 
                                                encoder_hidden_states = prompt_embeds, 
                                                down_block_additional_residuals = down_block_res_samples,
                                                mid_block_additional_residual = mid_block_res_sample,
                                                down_intrablock_additional_residuals = down_intrablock_additional_residuals, # Added for T2I adapter
                                                added_cond_kwargs = added_cond_kwargs,
                                                cross_attention_kwargs = {
                                                    'attn_mask': None,
                                                    'mask_ca': False,
                                                    'mask_sa': False
                                                },
                                                ).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                            pred_z0 = self.scheduler.step(noise_pred_uncond, t, x).pred_original_sample
                            pred_z0_cond = self.scheduler.step(noise_pred, t, x).pred_original_sample
                        else:
                            pred_z0 = self.scheduler.step(noise_pred, t, x).pred_original_sample

                        bg_loss = self.cal_bg_loss(pred_z0, image, mask)
                        if do_classifier_free_guidance:
                            if (self.cal_bg_loss(pred_z0, pred_z0_cond, torch.full_like(mask, 1)) != 0):
                                semantic_loss = self.cal_bg_loss(pred_z0, pred_z0_cond, mask) / self.cal_bg_loss(pred_z0, pred_z0_cond, torch.full_like(mask, 1))
                            else:
                                semantic_loss = 0
                        else:
                            semantic_loss = 0
                        loss = bg_loss + coef[t] * semantic_loss
                        print(f"loss: {loss.item()}")

                        loss_grad = torch.autograd.grad(
                            loss, x, retain_graph=False, create_graph=False
                        )[0].detach()
                        x_m = momentum * x_m - lr_xt[t] * loss_grad
                        x_m = x_m.to(self.unet.dtype)
                        x = x + x_m

        with torch.no_grad():
            x_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
            down_block_res_samples = None
            mid_block_res_sample = None
            if "controlnet" in model_list:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    x_input,
                    t,
                    encoder_hidden_states = prompt_embeds,
                    controlnet_cond = cond_image,
                    conditioning_scale = self.controlnet_scale,
                    return_dict = False,
                )
            down_intrablock_additional_residuals = None # Added for T2I adapter
            if "t2iadapter" in model_list:
                # Added for T2I adapter
                down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
            if do_classifier_free_guidance:
                if (t>=600):
                    noise_pred = self.unet(x_input,
                                        t, 
                                        encoder_hidden_states = prompt_embeds, 
                                        down_block_additional_residuals = down_block_res_samples,
                                        mid_block_additional_residual = mid_block_res_sample,
                                        down_intrablock_additional_residuals = down_intrablock_additional_residuals, # Added for T2I adapter
                                        added_cond_kwargs = added_cond_kwargs,
                                        cross_attention_kwargs = {
                                            'attn_mask': attention_mask,
                                            'mask_ca': True,
                                            'mask_sa': True,
                                        },
                                        ).sample
                else:
                    noise_pred = self.unet(x_input,
                                        t, 
                                        encoder_hidden_states = prompt_embeds, 
                                        down_block_additional_residuals = down_block_res_samples,
                                        mid_block_additional_residual = mid_block_res_sample,
                                        down_intrablock_additional_residuals = down_intrablock_additional_residuals, # Added for T2I adapter
                                        added_cond_kwargs = added_cond_kwargs,
                                        cross_attention_kwargs = {
                                            'attn_mask': attention_mask,
                                            'mask_ca': True,
                                            'mask_sa': False,
                                        },
                                        ).sample                        
            else:
                noise_pred = self.unet(x_input,
                                    t, 
                                    encoder_hidden_states = prompt_embeds, 
                                    down_block_additional_residuals = down_block_res_samples,
                                    mid_block_additional_residual = mid_block_res_sample,
                                    down_intrablock_additional_residuals = down_intrablock_additional_residuals, # Added for T2I adapter
                                    added_cond_kwargs = added_cond_kwargs,
                                    cross_attention_kwargs = {
                                        'attn_mask': None,
                                        'mask_ca': False,
                                        'mask_sa': False,
                                    },
                                    ).sample   
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
               
        return x, noise_pred


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        lr_f: str = "exp",
        momentum: float = 0.7,
        lr: float = 0.05,
        lr_warmup: float = 0.01,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        coef: float = 110,
        coef_f: str = "linear",
        cond_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        op_interval: int = 10,
        num_gradient_ops: int = 10,
        std: int = 1499,
        gamma: float = 0.5,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        model_list: List[str] = ["base"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            image (`PIL.Image` or `torch.FloatTensor`):
                The source image to be edited.
            mask (`PIL.Image` or `torch.FloatTensor`):
                The mask to be applied to the input image.f
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            lr_f (`str`):
                Specifies the type of scheduler for the learning rate to optimize latents.
            momentum (`float`):
                The momentum factor for stochastic gradient descent to optimize latents.
            lr (`float`):
                The learning rate for optimizing latents.
            lr_warmup (`float`):
                The learning rate for optimizing latents at timestep T.
                Since the gradient is ambiguous at timestep T, the learning rate usually should be smaller than the initial learning rate for the purpose of "warmup".
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            coef (`float`):
                Corresponds to the value of parameter lambda (λ) at timestep T, 
                as described in the PILOT paper: https://arxiv.org/abs/2407.08019.                
            coef_f (`str`):
                Specifies the type of scheduler for the parameter lambda (λ).
            cond_image (`PIL.Image` or `torch.FloatTensor`):
                An image for spatial controls, such as a Canny edge map.
            op_interval (`int`):
                Corresponds to the value of optimization interval (τ), 
                as described in the PILOT paper: https://arxiv.org/abs/2407.08019.
            num_gradient_ops (`int`):
                The number of gradient descent operations performed during each optimization timestep.
            std (`float`):
                Coefficient to scale the parameter lambda (λ).
            gamma (`float`):
                Parameter gamma used to balance image quality and sampling speed, 
                as described in the PILOT paper.
            ip_adapter_image (`PIL.Image` or `torch.FloatTensor`): Image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            model_list (`List[str]`): loaded models.
        Examples:
        
        Returns (`PIL.Image` or `torch.FloatTensor`)  
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if image is None:
            raise ValueError("`image` input cannot be undefined.")
        if mask is None:
            raise ValueError("`mask_image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = num_images_per_prompt
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) * num_images_per_prompt
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale >= 1.0
        print("do classifier free guidance: ",do_classifier_free_guidance)

        # 3. prepare controlnet / t2i-adapter/ ip-adapter input
        if isinstance(self.controlnet, ControlNetModel):
            if cond_image!=None:
                cond_image = self.prepare_image(
                    image=cond_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                cond_image=cond_image.to(self.controlnet.dtype).to("cuda")

        t2iadapter_input = []
        if isinstance(self.adapter, MultiAdapter):
            if cond_image!=None:
                for one_image in cond_image:
                    one_image = _preprocess_adapter_image(one_image, height, width)
                    one_image = one_image.to(device=device, dtype=self.adapter.dtype)
                    t2iadapter_input.append(one_image)
                t2iadapter_input = t2iadapter_input.to(self.adapter.dtype).to("cuda")
        if isinstance(self.adapter, T2IAdapter):
            if cond_image!=None:
                one_image = _preprocess_adapter_image(cond_image, height, width)
                one_image = one_image.to(device=device, dtype=self.adapter.dtype)
                t2iadapter_input = one_image
                t2iadapter_input = t2iadapter_input.to(self.adapter.dtype).to("cuda")
                
        adapter_state = None
        if t2iadapter_input!=[] and self.adapter:
            if isinstance(self.adapter, MultiAdapter):
                adapter_state = self.adapter(t2iadapter_input, self.t2i_scale)
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v
            else:
                adapter_state = self.adapter(t2iadapter_input)
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v * self.t2i_scale
            if num_images_per_prompt > 1:
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
            if do_classifier_free_guidance:
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = torch.cat([v] * 2, dim=0)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )
        
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        
        # 4. Encode input prompt, image embeds for IP-Adapter
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 5. Prepare masked image embedding and downsampled mask
        mask, image = prepare_mask_and_masked_image(image,mask)
        image = image.to(self.vae.dtype).to(self.device)
        image = self.encode_image(image,generator=generator)

        mask = F.interpolate(mask, (self.unet.config.sample_size, self.unet.config.sample_size), mode='nearest')
        mask = kornia.morphology.erosion(mask, torch.ones((3,3)))
        mask = mask.to(self.vae.dtype).to(self.device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device)
        self.scheduler.eta = eta
        timesteps = self.scheduler.timesteps.to(self.device)
        
        # 7. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.unet.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Adjust coef based on the area of the unmask portion.
        count_white = torch.sum(torch.eq(mask, 1)).item()
        if count_white==0:
            coef_scale=80
        else:
            coef_scale = std**2 / count_white**2
        coef = coef_scale * coef

        generator = generator[0] if isinstance(generator, list) else generator

        # 10 prepare attention mask
        attention_mask = {}
        for attn_size in [64,32,16,8]:  # create attention masks for multi-scale layers in unet
            attention_mask[str(attn_size**2)] = (F.interpolate(1-mask, (attn_size,attn_size), mode='bilinear'))[0,0,...].to(self.device)
            attention_mask[str(attn_size**2)][attention_mask[str(attn_size**2)] < 1] = 0
            if torch.all(attention_mask[str(attn_size**2)] == 0):
                attention_mask[str(attn_size**2)] = torch.ones_like(attention_mask[str(attn_size**2)])
        cross_attention_kwargs = {}
        cross_attention_kwargs["attn_mask"] = attention_mask
        # no need for inpainting
        if torch.all(mask == 0):
            attention_mask = None
            num_gradient_ops = 0

        # 11. Denoising loop
        with self.progress_bar(total = num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                no_op = True
                if (i % op_interval == 0):
                    no_op = False
                if (t < 1000 * (1-gamma)):
                    no_op=True
                    noise_source_latents = self.scheduler.add_noise(
                        image, torch.randn(latents.shape, generator=generator, device=device), t
                    ).to(latents.dtype)
                    latents = latents * (mask<=0.5) + noise_source_latents * (mask>0.5)
                latents, noise_pred = self.optimize_xt(x = latents, 
                                        image = image, mask = mask, t = t, cfg = guidance_scale, lr_f = lr_f, 
                                        momentum = momentum, lr = lr,
                                        no_op = no_op, coef = coef, coef_f = coef_f, 
                                        attention_mask = attention_mask, prompt_embeds = prompt_embeds,
                                        cond_image = cond_image, lr_warmup = lr_warmup, 
                                        num_gradient_ops = num_gradient_ops, adapter_state = adapter_state,
                                        model_list = model_list,
                                        added_cond_kwargs = added_cond_kwargs)
                
                result = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = result.prev_sample    
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                    
        # 12. decode latents into images (To save GPU memory, each latent is decoded individually.)
        image_list=[]
        for i in range(len(latents)):
            image_list.append(self.decode_latents(latents[i].unsqueeze(0), return_type='tensor'))
        image = torch.concat(image_list, dim = 0)

        if output_type == "pil":
            image = (image/2+0.5).clamp(0,1)
            image = image.cpu().permute(0,2,3,1).float().numpy()            # 10. Convert to PIL
            image_result = self.numpy_to_pil(image)
        else:
            image_result = image
            
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image_result
