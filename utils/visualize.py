from utils.image_processor import preprocess_image, tensor2PIL, mask4image
from PIL import Image

def whitemask4image(image, mask_tensor):
    mask_add = mask_tensor.clone()
    for i in range(3):
        mask_add[:, i, ...][mask_add[:, i, ...] == 0] = 0.99
        mask_add[:, i, ...][mask_add[:, i, ...] == 1] = 0
        mask_tensor[:, i, ...][mask_tensor[:, i, ...] == 0] = 0
    image_mask = (preprocess_image(image) + 1) / 2 * mask_tensor
    image_mask = image_mask + mask_add
    image_mask = 2 * image_mask - 1
    image_mask = tensor2PIL(image_mask)
    return image_mask

def t2i_visualize(image, mask_image, result_list, W=512, H=512):
    # process mask and image
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    image_mask = whitemask4image(image, mask_tensor)
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 2
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(result_list[i], (W, 0))
        new_image_list.append(new_image)
    return new_image_list

def ipa_visualize(image, mask_image, ip_image, result_list, W=512, H=512):
    # process mask and image
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    image_mask = whitemask4image(image, mask_tensor)
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 3
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(ip_image, (W, 0))
        new_image.paste(result_list[i], (W * 2, 0))
        new_image_list.append(new_image)
    return new_image_list

def spatial_visualize(image, mask_image, cond_image,result_list, W=512, H=512):
    # process mask and image
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    image_mask = whitemask4image(image, mask_tensor)
    
    mask_add=mask_tensor.clone()
    for i in range(3):
        mask_add[:,i,...][mask_add[:,i,...]==0]=0.99
        mask_add[:,i,...][mask_add[:,i,...]==1]=0
        mask_tensor[:,i,...][mask_tensor[:,i,...]==0]=0
    image_mask_tmp=(preprocess_image(image)+1)/2*mask_tensor
    image_mask_tmp=image_mask_tmp+mask_add
    image_mask_tmp=2*image_mask_tmp-1
    image_mask_tmp=tensor2PIL(image_mask_tmp)
    cond_image=mask4image(preprocess_image(cond_image),-preprocess_image(mask_image))
    image_draw=mask4image(preprocess_image(image_mask_tmp),-cond_image)
    image_draw=tensor2PIL(image_draw)
    
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 3
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(image_draw, (W, 0))
        new_image.paste(result_list[i], (W * 2, 0))
        new_image_list.append(new_image)

    return new_image_list

def ipa_visualize(image, mask_image, ip_image, result_list, W=512, H=512):
    # process mask and image
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    image_mask = whitemask4image(image, mask_tensor)
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 3
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(ip_image, (W, 0))
        new_image.paste(result_list[i], (W * 2, 0))
        new_image_list.append(new_image)
    return new_image_list

def ipa_spatial_visualize(image, mask_image, ip_image, cond_image, result_list, W=512, H=512):
    # process mask and image
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    mask_tensor = (preprocess_image(mask_image) + 1) / 2
    image_mask = whitemask4image(image, mask_tensor)
    
    mask_add=mask_tensor.clone()
    for i in range(3):
        mask_add[:,i,...][mask_add[:,i,...]==0]=0.99
        mask_add[:,i,...][mask_add[:,i,...]==1]=0
        mask_tensor[:,i,...][mask_tensor[:,i,...]==0]=0
    image_mask_tmp=(preprocess_image(image)+1)/2*mask_tensor
    image_mask_tmp=image_mask_tmp+mask_add
    image_mask_tmp=2*image_mask_tmp-1
    image_mask_tmp=tensor2PIL(image_mask_tmp)
    cond_image=mask4image(preprocess_image(cond_image),-preprocess_image(mask_image))
    image_draw=mask4image(preprocess_image(image_mask_tmp),-cond_image)
    image_draw=tensor2PIL(image_draw)
    
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 4
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(image_draw, (W, 0))
        new_image.paste(ip_image, (W * 2, 0))
        new_image.paste(result_list[i], (W * 3, 0))
        new_image_list.append(new_image)

    return new_image_list
