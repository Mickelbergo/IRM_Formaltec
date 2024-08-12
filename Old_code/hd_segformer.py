# -*- coding: utf-8 -*-
"""
Segformer HD model
"""

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torch.nn.functional as F
import math
import warnings
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

import config
DEVICE = config.DEVICE

model_version = "Segformer_pretrained_100"
md_model = torch.load(os.path.join(config.path, "runs", "Project", model_version, model_version
                        + "_last_epoch_model_fold_0.pth"), map_location = torch.device(DEVICE))

md_model.segmentation_head[1] = nn.UpsamplingBilinear2d(scale_factor = 1)

hd_model = smp.FPN(
    encoder_name = "mit_b2",
    encoder_weights = "imagenet",
    classes = 11,
    activation = "softmax")


# The model deciding which areas to focus on is not trained to ease training of the hd model
for parameter in md_model.parameters(): 
    parameter.requires_grad = False
    
#%%

# unterschiedliche Auflösung noch unberücksichtigt------------------------------------------
def reassemble(masks, hd_masks, index):
    b, c, h, w = masks.size()

    r1 = 100
    size = 512
    
    focalx = index%w*4 # downsample 4 during first model pass
    focaly = torch.div(index, w, rounding_mode="floor")*4
    
    index_t = get_hd_index(size, size, r1, size/2, size/2)
    index_im = get_hd_index(h, w, r1, focalx, focaly)
    
    masks[index_t[:,0], index_t[:,1], index_t[:,2]] = hd_masks[index_im[:,0], index_im[:,1], index_im[:,2]]
    
    return masks


def get_hd_index(h, w, r, x_c, y_c, b):
    
    x,y = torch.meshgrid(torch.arange(w).to(torch.float32), torch.arange(h).to(torch.float32))

    xx = x - x_c -0.001 # to avoid 0 radius
    yy = y - y_c -0.001

    radius = torch.sqrt(xx**2 + yy**2)

    radius[radius>r] = 0
    radius[radius>0] = 1

    radius = torch.tile(radius.unsqueeze(2), (1,1,3))

    indices = torch.nonzero(radius) 
    y_ind, x_ind, c_ind = indices[:,0], indices[:,1], indices[:,2]
    
    index = torch.Tensor([[c_ind[i], y_ind[i], x_ind[i]] for i in range(len(c_ind))]).to(torch.long)
    
    index = torch.tile(index.unsqueeze(0), (b,1,1,1))

    return index

def get_hd_index_im(h, w, r, x_c, y_c):
    
    x,y = torch.meshgrid(torch.arange(w).to(torch.float32), torch.arange(h).to(torch.float32))

    x = torch.tile(x.unsqueeze(0), (len(x_c),1,1)) # b * h * w
    y = torch.tile(y.unsqueeze(0), (len(x_c),1,1))
    
    xx = x - x_c -0.001 # to avoid 0 radius
    yy = y - y_c -0.001

    radius = torch.sqrt(xx**2 + yy**2)

    radius[radius>r] = 0
    radius[radius>0] = 1

    radius = torch.tile(radius.unsqueeze(3), (1,1,1,3))

    indices = torch.nonzero(radius) 
    b_ind, y_ind, x_ind, c_ind = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    
    index = torch.Tensor([[b_ind[i], c_ind[i], y_ind[i], x_ind[i]] for i in range(len(c_ind))]).to(torch.long)

    return index


def get_md_index(h, w, r1, r2, x_c, y_c, x_ct, y_ct, f):
    
    x,y = torch.meshgrid(torch.arange(w).to(torch.float32), torch.arange(h).to(torch.float32))
    
    x = torch.tile(x.unsqueeze(0), (len(x_c),1,1)) # b * h * w
    y = torch.tile(y.unsqueeze(0), (len(x_c),1,1))
    
    xx = x - x_ct -0.001 # to avoid 0 radius
    yy = y - y_ct -0.001

    radius = torch.sqrt(xx**2 + yy**2)
    rad = torch.sqrt(xx**2 + yy**2)

    radius[radius>(r2+r1)] = 0
    radius[radius<(r1)] = 0
    radius[radius>0] = 1
    
    radius2 = f * torch.sqrt(xx**2 + yy**2) - (r1+r2) # increased radius, because image pixels are averaged

    radius = torch.tile(radius.unsqueeze(3), (1,1,1,3))

    indices = torch.nonzero(radius) 
    b_ind, y_ind, x_ind, c_ind = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    
    ind_rad = torch.Tensor([[b_ind[i], y_ind[i], x_ind[i]] for i in range(len(x_ind))]).to(torch.long)
    
    x_ind = x_ind - x_ct
    y_ind = y_ind - y_ct
    
    rel_increase = (radius2[ind_rad[:,:,0],ind_rad[:,:,1]]/rad[ind_rad[:,:,0],ind_rad[:,:,1]])
    x_ind = x_ind * rel_increase
    y_ind = y_ind * rel_increase
    
    x_ind = x_ind + x_c
    y_ind = y_ind + y_c
    index = torch.Tensor([[b_ind[i], c_ind[i], y_ind[i], x_ind[i]] for i in range(len(c_ind))]).to(torch.long)

    return index


def get_index_target(h, w, r1, r2, x_c, y_c, b):
    
    x,y = torch.meshgrid(torch.arange(w).to(torch.float32), torch.arange(h).to(torch.float32))

    xx = x - x_c -0.001 # to avoid 0 radius
    yy = y - y_c -0.001

    radius = torch.sqrt(xx**2 + yy**2)

    radius[radius>(r2+r1)] = 0
    radius[radius<r1] = 0
    radius[radius>0] = 1

    radius = torch.tile(radius.unsqueeze(2), (1,1,3))
    
    indices = torch.nonzero(radius) 
    y_ind, x_ind, c_ind = indices[:,0], indices[:,1], indices[:,2]

    index = torch.Tensor([[c_ind[i], y_ind[i], x_ind[i]] for i in range(len(c_ind))]).to(torch.long)
    
    index = torch.tile(index.unsqueeze(0), (b,1,1,1))
    
    return index

def retina_transform(index, image, h, w, size):
    '''
    transform hd image to smaller image with sharp center and blurry periphery
    '''
    image = image.to("cpu") # maybe not necessary on larger GPU--------------------
    b = image.size()[0]
    
    # how much padding needed?-----------------------------------
    padding = torchvision.transforms.Pad(padding=(2048, 2048, 2048, 2048), 
                                         fill= 0, padding_mode = "constant")

    # index tensor of length batch
    focalx = 2048 + index%w*4*4 # 4 times higher res, downsample 4 during first model pass
    focaly = 2048 + torch.div(index, w, rounding_mode="floor")*4*4
    
    r1 = 100 # Radius with full resolution
    r2 = 80 # Radius with average 2x2 pixel resolution
    r3 = 70 # Radius with average 3x3 pixel resolution
    target_im_size = [b, 3, size, size]

    av2 = nn.AvgPool2d(kernel_size=(2,2), stride=1)
    im_av2 = padding(av2(image))
    
    av3 = nn.AvgPool2d(kernel_size=(3,3), stride=1)
    im_av3 = padding(av3(image))
    #plt.imshow(im_av3.type(torch.int16).permute(1,2,0))
    #plt.show()
    
    av4 = nn.AvgPool2d(kernel_size=(4,4), stride=1)
    im_av4 = padding(av4(image))
    
    image = padding(image)
    
    retina_im = torch.zeros(target_im_size)

    th = target_im_size[2] # target hight
    tw = target_im_size[3] # target width
    
    index_t = get_hd_index(th, tw, r1, th/2, tw/2, b)
    index_im = get_hd_index_im(image.size()[2], image.size()[3], r1, focalx, focaly)
    retina_im[index_t[:,0], index_t[:,1], index_t[:,2]] = image[index_im[:,0], index_im[:,1], index_im[:,2]]
    
    index_t = get_index_target(th, tw, r1, r2, th/2, tw/2, b)
    index_im = get_md_index(th, tw, r1, r2, focalx, focaly, th/2, tw/2, 2)
    retina_im[index_t[:,0], index_t[:,1], index_t[:,2]] = im_av2[index_im[:,0], index_im[:,1], index_im[:,2]]
    
    index_t = get_index_target(th, tw, r1+r2, r3, th/2, tw/2, b)
    index_im = get_md_index(th, tw, r1+r2, r3, focalx, focaly, th/2, tw/2, 3)
    retina_im[index_t[:,0], index_t[:,1], index_t[:,2]] = im_av3[index_im[:,0], index_im[:,1], index_im[:,2]]
    
    index_t = get_index_target(th, tw, r1+r2+r3, (torch.sqrt(th**2 + tw**2)/2)-(r1+r2+r3), th/2, tw/2, b)
    index_im = get_md_index(th, tw, r1+r2+r3, (torch.sqrt(th**2 + tw**2)/2)-(r1+r2+r3), focalx, focaly, th/2, tw/2, 4)
    retina_im[index_t[:,0], index_t[:,1], index_t[:,2]] = im_av4[index_im[:,0], index_im[:,1], index_im[:,2]]
    
    return retina_im

class HD_Segformer(nn.Module):
    def __init__(
        self,
        first_model,
        second_model,
        n_cls,
        patch_size,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.first_model = first_model
        self.second_model = second_model
        self.softmax = nn.Softmax(dim = 1)  
        self.retina_transform = retina_transform
        self.reassemble = reassemble

    def forward(self, im, im_large):

        H, W = im.size(2), im.size(3)

        first_masks = self.first_model(im)

        b, c, h, w = first_masks.size()

        masks = torch.sum(first_masks[:, 1:], dim=1) # add all non-background channels 

        # select index of patch with the most certain non-background prediction
        # This will be the centre of the hd image
        masks = torch.flatten(masks, start_dim=1)
        index = torch.argmax(masks, dim=1) 

        hd_im = self.retina_transform(index, im_large, h, w, 512)
        
        hd_masks = self.second_model(hd_im)
        
        masks = self.reassemble(first_masks, hd_masks, index)
        
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        masks = self.softmax(masks)

        return masks


#%%

model_cfg = {
    "patch_size": 16, 
    "n_cls": 11, # number of classes + 1
}


hd_vit_model = HD_Segformer(md_model, hd_model, n_cls=model_cfg["n_cls"], 
                            patch_size=model_cfg["patch_size"])

#%%
x = torch.rand((8,11,128,128))
