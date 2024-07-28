import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, pipeline
import torch
from controlnet_aux import HEDdetector, LineartDetector, OpenposeDetector

palette = np.asarray([[0, 0, 0],[120, 120, 120],[10, 120, 120],[6, 230, 230],[80, 50, 50],[4, 200, 3],[120, 120, 80],[140, 140, 140],[204, 5, 255],[230, 230, 230],[4, 250, 7],[224, 5, 255],[235, 255, 7],[150, 5, 61],
    [120, 120, 70],[8, 255, 51],[255, 6, 82],[143, 255, 140],[204, 255, 4],[255, 51, 7],[204, 70, 3],[0, 102, 200],[61, 230, 250],[255, 6, 51],[11, 102, 255],[255, 7, 71],[255, 9, 224],[9, 7, 230],[220, 220, 220],
    [255, 9, 92],[112, 9, 255],[8, 255, 214],[7, 255, 224],[255, 184, 6],[10, 255, 71],[255, 41, 10],[7, 255, 255],[224, 255, 8],[102, 8, 255],[255, 61, 6],[255, 194, 7],[255, 122, 8],[0, 255, 20],[255, 8, 41],[255, 5, 153],
    [6, 51, 255],[235, 12, 255],[160, 150, 20],[0, 163, 255],[140, 140, 140],[250, 10, 15],[20, 255, 0],[31, 255, 0],[255, 31, 0],[255, 224, 0],[153, 255, 0],[0, 0, 255],[255, 71, 0],[0, 235, 255],[0, 173, 255],[31, 0, 255],[11, 200, 200],
    [255, 82, 0],[0, 255, 245],[0, 61, 255],[0, 255, 112],[0, 255, 133],[255, 0, 0],[255, 163, 0],[255, 102, 0],[194, 255, 0],[0, 143, 255],[51, 255, 0],[0, 82, 255],[0, 255, 41],[0, 255, 173],[10, 0, 255],[173, 255, 0],[0, 255, 153],
    [255, 92, 0],[255, 0, 255],[255, 0, 245],[255, 0, 102],[255, 173, 0],[255, 0, 20],[255, 184, 184],[0, 31, 255],[0, 255, 61],[0, 71, 255],[255, 0, 204],[0, 255, 194],[0, 255, 82],[0, 10, 255],[0, 112, 255],[51, 0, 255],[0, 194, 255],
    [0, 122, 255],[0, 255, 163],[255, 153, 0],[0, 255, 10],[255, 112, 0],[143, 255, 0],[82, 0, 255],[163, 255, 0],[255, 235, 0],[8, 184, 170],[133, 0, 255],[0, 255, 92],[184, 0, 255],[255, 0, 31],[0, 184, 255],[0, 214, 255],[255, 0, 112],
    [92, 255, 0],[0, 224, 255],[112, 224, 255],[70, 184, 160],[163, 0, 255],[153, 0, 255],[71, 255, 0],[255, 0, 163],[255, 204, 0],[255, 0, 143],[0, 255, 235],[133, 255, 0],[255, 0, 235],[245, 0, 255],[255, 0, 122],[255, 245, 0],[10, 190, 212],
    [214, 255, 0],[0, 204, 255],[20, 0, 255],[255, 255, 0],[0, 153, 255],[0, 41, 255],[0, 255, 204],[41, 0, 255],[41, 255, 0],[173, 0, 255],[0, 245, 255],[71, 0, 255],[122, 0, 255],[0, 255, 184],[0, 92, 255],[184, 255, 0],[0, 133, 255],
    [255, 214, 0],[25, 194, 194],[102, 255, 0],[92, 0, 255],])


def img2canny(image_cond):
    image_cond = np.array(image_cond)
    low_threshold = 100
    high_threshold = 200
    image_cond = cv2.Canny(image_cond, low_threshold, high_threshold)
    image_cond = image_cond[:, :, None]
    image_cond = np.concatenate([image_cond, image_cond, image_cond], axis=2)
    image_cond = Image.fromarray(image_cond)
    return image_cond

def img2seg(image_cond,model_path):
    image_processor = AutoImageProcessor.from_pretrained(f"{model_path}/upernet-convnext-large")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(f"{model_path}/upernet-convnext-large")
    pixel_values = image_processor(image_cond, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image_cond.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    image_cond = Image.fromarray(color_seg)
    return image_cond

def img2scribble(image_cond, model_path):
    hed = HEDdetector.from_pretrained(f"{model_path}/ControlNet")
    image_cond = hed(image_cond, scribble=True)
    return image_cond

def img2normal(image_cond,model_path):
    W, H = image_cond.size
    depth_estimator = pipeline("depth-estimation", model =f"{model_path}/dpt-hybrid-midas" )
    image_cond = depth_estimator(image_cond)['predicted_depth'][0]
    image_cond = image_cond.numpy()
    image_depth = image_cond.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    bg_threhold = 0.4
    x = cv2.Sobel(image_cond, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0
    y = cv2.Sobel(image_cond, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image_cond = np.stack([x, y, z], axis=2)
    image_cond /= np.sum(image_cond ** 2.0, axis=2, keepdims=True) ** 0.5
    image_cond = (image_cond * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image_cond = Image.fromarray(image_cond)
    image_cond = image_cond.resize((W, H), Image.NEAREST)
    return image_cond

def img2lineart(image_cond, model_path):
    processor = LineartDetector.from_pretrained(f"{model_path}/ControlNet")
    image_cond = processor(image_cond)
    return image_cond

def img2openpose(image_cond, model_path):
    processor = OpenposeDetector.from_pretrained(
        f"{model_path}/sd-controlnet-v11-openpose"
    )
    image_cond = processor(image_cond, hand_and_face=True)
    return image_cond

def img2tile(image_cond, block_size=8):
    if isinstance(image_cond, str):
        image = Image.open(image_cond)
    else:
        image = image_cond
    
    color_palette = image.resize((block_size, block_size))
    color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)
    return color_palette

def img2cond(controlnet_id, image, model_path):
    if "canny" in controlnet_id:
        return img2canny(image)
    elif "seg" in controlnet_id:
        return img2seg(image, model_path)
    elif "scribble" in controlnet_id or "sketch" in controlnet_id:
        return img2scribble(image, model_path)
    elif "normal" in controlnet_id:
        return img2normal(image, model_path)
    elif "lineart" in controlnet_id:
        return img2lineart(image,model_path)
    elif "openpose" in controlnet_id:
        return img2openpose(image, model_path)
    elif "tile" in controlnet_id:
        return img2tile(image, block_size=8)
    else:
        raise ValueError("Invalid control condition")