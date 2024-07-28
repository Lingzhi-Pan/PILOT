import torch
import numpy as np
from PIL import Image

# PIL2tensor
def preprocess_image(image):
    image = np.array(image).astype(np.float32) / 255.0
    # # b,c,w,h
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).clamp(0, 1)
    return 2 * image - 1

def tensor2PIL(image):
    image = image.squeeze(0)
    image = ((image + 1.0) / 2.0 * 255.0).permute(1, 2, 0).cpu().numpy().astype(int)
    image = Image.fromarray(np.uint8(image))
    return image

def mask4image(image, mask):
    image = (image + 1) / 2
    mask = (mask + 1) / 2
    image = image * (mask > 0.5)
    image = 2 * image - 1
    return image