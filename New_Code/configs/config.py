'''
Parameters used for training the model are defined in this file. 
'''

import torch
# import os
# import json
import numpy as np
# import segmentation_models_pytorch as smp

if torch.cuda.is_available():
    DEVICE = 0 #"cpu" 
else:
    DEVICE = "cpu"
    
path = r"E:\ForMaLTeC\Wound_segmentation_III\Data" 



#################################

ENCODER = "mit_b4"  # "se_resnext50_32x4d" "mit_b2" "hd_model" "segmenter"
DECODER = "FPN" # "UNET"
ACTIVATION = "softmax2d" # "softmax2d"
BATCHSIZE = 8
model_version = f"640_{DECODER}_{ENCODER}_{ACTIVATION}_no_classes_label_smoothing"
ENCODER_WEIGHTS = "imagenet"


#######################







use_decoder = True # Needs to be true for UNET, FPN, "se_resnext50_32x4d", "mit_b2", ...-> all models from segmentation models pytorch


# wound areas used in the loss function (pixels_per_class/total_pixels)
areas = np.array([1, 0.02])

# Define wound classes (die Reihenfolge ist die gleiche, wie bei der Erstellung der Masken)
wound_classes = ["background", 'wound']





# # Because of large class imbalances these are not very reliable, but give a rough estimate of the quality of the predictions
# metrics = [
#      smp.utils.metrics.IoU(threshold=0.5),
#      smp.utils.losses.DiceLoss(),
# ]








optimizer_lr = 0.0001



# exponential decay parameter in the learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
lr_scheduler_gamma = 0.98

loss = "custom_weighted_BCE" # "sqrt_custom_weighted_BCE" 
# Code needs to be adapted to use "weighted_BCE", "sqrt_weighted_BCE", "weighted_FTL", "weighted_FTL_BCE" -----------------------------------------