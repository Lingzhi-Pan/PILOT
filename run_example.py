from PIL import Image, ImageOps
import numpy as np
import torch
import argparse
import os
from omegaconf import OmegaConf
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    T2IAdapter,
)
from pipeline.pipeline_pilot import PilotPipeline, AutoencoderKL
from models.attn_processor import revise_pilot_unet_attention_forward
import os
import torch.nn.functional as F
from utils.generate_spatial_map import img2cond
from utils.image_processor import preprocess_image, tensor2PIL, mask4image
from utils.visualize import t2i_visualize, spatial_visualize, ipa_visualize, ipa_spatial_visualize
from daam import trace, set_seed
import matplotlib.pyplot as plt
# import xformers

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", type=str, default="coco1.yaml"
)

args = parser.parse_args() 
config = OmegaConf.load(args.config_file)

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

prompt_list = [config.prompt]
negative_prompt = [config.negative_prompt]
device = "cuda"
controlnet = None
adapter = None

model_list = ["base"]

if config.fp16:
    weight_format = torch.float16
else:
    weight_format = torch.float32

image = Image.open(config.input_image).convert("RGB")
image = image.resize((config.W, config.H), Image.NEAREST)
mask_image = Image.open(config.mask_image).convert("RGB")
mask_image = mask_image.resize((config.W, config.H), Image.NEAREST)
if mask_image.mode != "RGB":
    mask_image = mask_image.convert("RGB")
for x in range(config.W):
    for y in range(config.H):
        r, g, b = mask_image.getpixel((x, y))
        if (r, g, b) != (0, 0, 0) and (r, g, b) != (255, 255, 255):
            mask_image.putpixel((x, y), (0, 0, 0))

################################### loading models and additional controls #############################
# load controlnet
if "controlnet_id" in config:
    print("load controlnet")
    model_list.append("controlnet")
    controlnet = ControlNetModel.from_pretrained(
        f"{config.model_path}/{config.controlnet_id}", torch_dtype = weight_format
    ).to(device)

# load t2i adapter
if "t2iadapter_id" in config:
    print("load t2i adapter") 
    model_list.append("t2iadapter")
    adapter = T2IAdapter.from_pretrained(
        f"{config.model_path}/{config.t2iadapter_id}", torch_dtype = weight_format
    ).to(device)

# process spatial controls
cond_image = None
if ("controlnet" in model_list) or ("t2iadapter" in model_list):
    print("process spatial controls")
    spatial_id = config.controlnet_id if "controlnet_id" in config else config.t2iadapter_id
    cond_image = Image.open(config.cond_image).convert("RGB")
    cond_image = cond_image.resize((config.W, config.H), Image.NEAREST) 
    cond_image = img2cond(spatial_id ,cond_image, config.model_path)
    cond_image = ImageOps.invert(cond_image)
    cond_image.save("cond.png")
    image_convert = img2cond(spatial_id, image, config.model_path)
    image_convert = mask4image(
        -preprocess_image(image_convert), preprocess_image(mask_image)
    )
    cond_image = mask4image(
        -preprocess_image(cond_image), -preprocess_image(mask_image)
    )
    cond_image = (image_convert + 1) / 2 + (cond_image + 1) / 2
    cond_image = 2 * cond_image - 1
    cond_image = tensor2PIL(-cond_image)

# load base model
print("load base model")
if config.fp16:
    vae = AutoencoderKL.from_pretrained(
        f"{config.model_path}/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=torch.float16,
        requires_safety_checker=False,
    )
    pipe = PilotPipeline.from_pretrained(
        f"{config.model_path}/{config.model_id}",
        vae=vae,
        controlnet=controlnet,
        adapter=adapter,
        torch_dtype=torch.float16,
        variant="fp16",
        requires_safety_checker=False,
    ).to(device)
    # pipe.save_pretrained(f"{config.model_path}/disney_style")
else:
    vae = AutoencoderKL.from_pretrained(
        f"{config.model_path}/stable-diffusion-v1-5",
        subfolder="vae",
        requires_safety_checker=False,
    )
    pipe = PilotPipeline.from_pretrained(
        f"{config.model_path}/{config.model_id}",
        vae=vae,
        controlnet=controlnet,
        adapter=adapter,
        requires_safety_checker=False,
    ).to(device)
    
if "t2iadapter" in model_list:
    if "t2iadapter_scale" in config:
        pipe.set_t2i_adapter_scale([config.t2iadapter_scale])
    else:
        pipe.set_t2i_adapter_scale(1)
        
if "controlnet" in model_list:
    if "controlnet_scale" in config:
        pipe.set_controlnet_scale([config.controlnet_scale])
    else:
        pipe.set_controlnet_scale(1)

# load lora
if "lora_id" in config:
    print("load lora")
    for i in range(len(config.lora_id)):
        lora_id = config.lora_id[i]
        lora_scale = config.lora_scale[i]
        pipe.load_lora_weights(
            f"{config.model_path}/{lora_id}",
            weight_name="model.safetensors",
            torch_dtype=torch.float16,
            adapter_name=lora_id,
        )
        print(f"lora id: {lora_id}*{lora_scale}")
        pipe.set_adapters(lora_id, adapter_weights=lora_scale)

# load ip adapter
ip_image = None
if "ipa_id" in config:
    print("load ip adapter")
    model_list.append("ipa")
    pipe.load_ip_adapter(
        f"{config.model_path}/ip_adapter",
        subfolder="v1-5",
        weight_name="ip-adapter_sd15_light.bin",
    )
    revise_pilot_unet_attention_forward(pipe.unet)

    ip_image = Image.open(config.ip_image)
    ip_image = ip_image.resize((config.W, config.H), Image.NEAREST)
    if "ip_scale" not in config:
        config.ip_scale = 0.8
    pipe.set_ip_adapter_scale(config.ip_scale)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator(device="cuda").manual_seed(config.seed)
# pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda", weight_format)
# print("unet: ",pipe.unet)
#################################### run examples and save results ##########################
image_list = pipe(
    prompt=prompt_list,
    negative_prompt=negative_prompt,
    num_inference_steps=config.step,
    height=config.H,
    width=config.W,
    guidance_scale=config.cfg,
    num_images_per_prompt = config.num,
    image=image,
    mask=mask_image,
    generator=generator,
    lr_f=config.lr_f,
    momentum=config.momentum,
    lr=config.lr,
    lr_warmup=config.lr_warmup,
    coef=config.coef,
    coef_f=config.coef_f,
    op_interval=config.op_interval,
    cond_image=cond_image,
    num_gradient_ops=config.num_gradient_ops,
    gamma=config.gamma,
    return_dict=True,
    ip_adapter_image=ip_image,
    model_list=model_list,
    inter_save=False,
)

if "ipa" in model_list and "controlnet" in model_list:
    new_image_list = ipa_spatial_visualize(image=image, mask_image=mask_image, ip_image=ip_image, cond_image=cond_image, result_list=image_list)
elif "controlnet" in model_list or "t2iadapter" in model_list:
    new_image_list = spatial_visualize(image=image, mask_image=mask_image, cond_image=cond_image, result_list=image_list)
elif "ipa" in model_list:
    new_image_list = ipa_visualize(image=image, mask_image=mask_image, ip_image=ip_image, result_list=image_list)
else:
    new_image_list = t2i_visualize(image=image, mask_image=mask_image, result_list=image_list)


file_path = (
    f"{config.output_path}/seed{config.seed}_step{config.step}.png"
)
for new_image in new_image_list:
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file_path)
        j = 0
        while True:
            j += 1
            file_path = f"{base}_{j}{ext}"
            if not os.path.exists(file_path):
                break
    new_image.save(file_path)
    print(f"image save in {file_path}")
