'''
Define the training augmentation.
'''
import torch
import torch.nn as nn
import kornia as K


class training_augmentation_imlarge(nn.Module):
    
  def __init__(self):
    super(training_augmentation_imlarge, self).__init__()

    self.k1 = K.augmentation.RandomHorizontalFlip(p=0.5)
    self.k2 = K.augmentation.RandomVerticalFlip(p=0.2)
    self.k3 = K.augmentation.RandomAffine(degrees = 10, translate=0.1, 
                                scale=(0.7, 1.8), resample="nearest", p = 1)
    self.k4 = K.augmentation.RandomCrop(size=(512, 512), p = 1)
    self.k5 = K.augmentation.RandomCrop(size=(2048, 2048), p = 1)
  
  def forward(self, img: torch.Tensor, img_large: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    img_out = self.k4(self.k3(self.k2(self.k1(img))))

    mask_out = self.k4(self.k3(self.k2(self.k1(mask, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
    weights_out = self.k4(self.k3(self.k2(self.k1(weights, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
    h, w, c = img_large.size()  
    self.k3._params["center"] *= 4
    self.k3._params["translations"] *= 4
    self.k4._params["input_size"] = [[h, w]]
    self.k4._params["dst"] = torch.Tensor([[[0., 0.], [2047., 0.], [2047., 2047.], [0., 2047.]]])
    self.k4._params["src"] = self.k4._params["dst"] + self.k4._params["src"][0][0]*4
    img_large_out = self.k5(self.k3(self.k2(self.k1(img_large, self.k1._params),  self.k2._params),  self.k3._params), self.k4._params)

    return img_out[0], img_large_out[0], mask_out[0], weights_out[0]


# probability set to 0 for all augs but random crop. Crop necessary because of positional encoding
class valid_augmentation_imlarge(nn.Module):
    
  def __init__(self):
    super(valid_augmentation_imlarge, self).__init__()

    self.k1 = K.augmentation.RandomHorizontalFlip(p=0)
    self.k2 = K.augmentation.RandomVerticalFlip(p=0)
    self.k3 = K.augmentation.RandomAffine(degrees = 10, translate=0.1, scale=(0.7, 1.8), p = 0)
    self.k4 = K.augmentation.RandomCrop(size=(512, 512), p = 1)
    self.k5 = K.augmentation.RandomCrop(size=(2048, 2048), p = 1)
  
  def forward(self, img: torch.Tensor, img_large: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    img_out = self.k4(self.k3(self.k2(self.k1(img))))

    mask_out = self.k4(self.k3(self.k2(self.k1(mask, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
    weights_out = self.k4(self.k3(self.k2(self.k1(weights, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
    h, w, c = img_large.size()  
    self.k3._params["center"] *= 4
    self.k3._params["translations"] *= 4
    self.k4._params["input_size"] = [[h, w]]
    self.k4._params["dst"] = torch.Tensor([[[0., 0.], [2047., 0.], [2047., 2047.], [0., 2047.]]])
    self.k4._params["src"] = self.k4._params["dst"] + self.k4._params["src"][0][0]*4
    img_large_out = self.k5(self.k3(self.k2(self.k1(img_large, self.k1._params),  self.k2._params),  self.k3._params), self.k4._params)

    return img_out[0], img_large_out[0], mask_out[0], weights_out[0]


class training_augmentation(nn.Module):

  def __init__(self):
    super(training_augmentation, self).__init__()

    self.k1 = K.augmentation.RandomHorizontalFlip(p=0.5)
    self.k2 = K.augmentation.RandomVerticalFlip(p=0.2)
    self.k3 = K.augmentation.RandomAffine(degrees = 30, translate=0.1, scale=(0.6, 1.5), resample="nearest", p = 1)
    self.k4 = K.augmentation.RandomCrop(size=(512,512), p = 1)
    self.k5 = K.augmentation.RandomBrightness(brightness=(0.85, 1.15), p=1)
    self.k6 = K.augmentation.RandomContrast(contrast=(0.85, 1.15), p=1)
    self.k7 = K.augmentation.RandomSharpness(sharpness=0.5, p=1)
    # self.k8 = K.augmentation.RandomGamma(gamma=(0.8, 1.2))
    
  def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    img_out = self.k7(self.k6(self.k5(self.k4(self.k3(self.k2(self.k1(img))))/255.0)))*255.0 # some augmentations need range (0,1)
    mask_out = self.k4(self.k3(self.k2(self.k1(mask, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    # weights_out = self.k4(self.k3(self.k2(self.k1(weights, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)

    return img_out[0], mask_out[0] #, weights_out[0]


# class valid_augmentation(nn.Module):
    
#   def __init__(self):
#     super(valid_augmentation, self).__init__()

#     self.k1 = K.augmentation.RandomHorizontalFlip(p=0.5)
#     self.k2 = K.augmentation.RandomVerticalFlip(p=0.2)
#     self.k3 = K.augmentation.RandomAffine(degrees = 10, translate=0.1, scale=(0.7, 1.8), p = 1)
#     self.k4 = K.augmentation.RandomCrop(size=(512,512), p = 1)
  
#   def forward(self, img: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

#     img_out = self.k4(self.k3(self.k2(self.k1(img))))

#     mask_out = self.k4(self.k3(self.k2(self.k1(mask, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
#     weights_out = self.k4(self.k3(self.k2(self.k1(weights, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)

#     return img_out[0], mask_out[0], weights_out[0]


'''
# Alternative using albumentations. Training on GPU with Kornia is faster. 

import albumentations as albu

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),

        albu.ShiftScaleRotate(scale_limit=(-0.3, 0.8), rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True)
    ]

    return albu.Compose(train_transform)
'''
