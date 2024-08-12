'''
A collection of weighted loss functions based on the Binary Cross-Entropy and Focal Tversky Loss functions. 
Note that some use inputs, targets and pixel_weights as input, while others only use inputs and targets. 
(-> Adjustments needed to switch between the two -------------------------------)
'''

# import necessary libraries
from configs import config

import numpy as np
import torch
import torch.nn as nn

DEVICE = config.DEVICE
# wound areas used in the loss function (pixels_per_class/total_pixels)
areas = config.areas


class custom_weighted_BCE(nn.Module):
    def __init__(self):
        super(custom_weighted_BCE, self).__init__()
        self._name = "custom_weighted_BCE"
        weights = torch.tensor(1/areas) # without nichttraumatische Hautveränderungen
        self.weights = weights.to(DEVICE)


    def forward(self, inputs, targets):
        
        epsilon = 0.1
        targets = (1.0-epsilon) * targets + epsilon/targets.size(1)

        reduce_axis = list(range(2, len(inputs.shape)))
        
        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.weights * torch.mean((targets * torch.log(inputs + eps)), axis=reduce_axis) \
            - 5 * torch.mean(((1 - targets) * torch.log(1 - inputs + eps)), axis=reduce_axis)
        BCE = BCE.mean()/10

        return BCE
      
# Loss function: weighted BCE including pixelwise weights according to the subjective certainty of classification during labeling
class custom_weighted_BCE_alt(nn.Module):
    def __init__(self):
        super(custom_weighted_BCE_alt, self).__init__()
        self._name = "custom_weighted_BCE_alt"
        weights = torch.tensor(1/areas[:-1]) # without nichttraumatische Hautveränderungen
        self.weights = weights.to(DEVICE)


    def forward(self, inputs, targets, pixel_weights):

        reduce_axis = list(range(2, len(inputs.shape)))
        
        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.weights * torch.mean(pixel_weights * (targets * torch.log(inputs + eps)), axis=reduce_axis) \
            - torch.mean(pixel_weights * ((1 - targets) * torch.log(1 - inputs + eps)), axis=reduce_axis)
        BCE = BCE.mean()

        return BCE
    
      

# Loss function: Square-root weighted BCE including pixelwise weights according to the subjective certainty of classification during labeling
class sqrt_custom_weighted_BCE(nn.Module):
    def __init__(self):
        super(sqrt_custom_weighted_BCE, self).__init__()
        self._name = "sqrt_custom_weighted_BCE"
        weights = torch.tensor(np.sqrt(0.01/areas[2:])) # without Hautrötung and background
        self.weights = weights.to(DEVICE)


    def forward(self, inputs, targets, pixel_weights):

        reduce_axis = list(range(2, len(inputs.shape)))
        
        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.weights * torch.mean(pixel_weights * (targets * torch.log(inputs + eps)), axis=reduce_axis) \
            - torch.mean(pixel_weights * ((1 - targets) * torch.log(1 - inputs + eps)), axis=reduce_axis)
        BCE = BCE.mean()

        return BCE

  
# Loss function: Weighted Binary Cross-Entropy Loss (BCE)
class weighted_BCE(nn.Module):
    def __init__(self):
        super(weighted_BCE, self).__init__()
        self._name = "weighted_BCE"
        weight = torch.tensor(0.01/areas[2:]) # without Hautrötung and background
        self.bce_weights = weight.to(DEVICE)
        

    def forward(self, inputs, targets_bce):

        reduce_axis = list(range(2, len(inputs.shape)))

        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.bce_weights * torch.mean(targets_bce * torch.log(inputs + eps), axis=reduce_axis) \
            - torch.mean((1 - targets_bce) * torch.log(1 - inputs + eps), axis=reduce_axis)
        BCE = BCE.mean()

        return BCE
    
    
# Loss function: Square-root weighted BCE
class sqrt_weighted_BCE(nn.Module):
    def __init__(self):
        super(sqrt_weighted_BCE, self).__init__()
        self._name = "sqrt_weighted_BCE"
        weight = torch.tensor(0.01/areas[2:]) # without Hautrötung and background
        self.bce_weights = weight.to(DEVICE)
        

    def forward(self, inputs, targets_bce):

        reduce_axis = list(range(2, len(inputs.shape)))

        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.bce_weights * torch.mean(targets_bce * torch.log(inputs + eps), axis=reduce_axis) \
            - torch.mean((1 - targets_bce) * torch.log(1 - inputs + eps), axis=reduce_axis)
        BCE = BCE.mean()

        return BCE
    
    

# Loss function: Weighted Focal Tversky Loss (FTL)
class weighted_FTL(nn.Module):
    def __init__(self):
        super(weighted_FTL, self).__init__()
        self._name = "weighted_FTL"
        weight = torch.tensor(0.01/areas[2:]) # without Hautrötung and background
        self.weights = weight.to(DEVICE)
        self.beta = 0.3
        self.alpha = 1-self.beta
        self.gamma = 1.3
        

    def forward(self, inputs, targets, smooth=1):

        reduce_axis = list(range(2, len(inputs.shape)))

        # True Positives, False Positives & False Negatives
        TP = torch.sum(targets * inputs, axis=reduce_axis)
        FP = torch.sum((1-targets) * inputs, axis=reduce_axis)
        FN = torch.sum(targets * (1-inputs), axis=reduce_axis)

        Tversky = (TP / (TP  + self.alpha*FN + self.beta*FP + smooth))

        FocalTversky = torch.mean(self.weights * (1 - Tversky)**self.gamma)

        return FocalTversky
    

# Loss function: Weighted FTL and BCE loss function
class weighted_FTL_BCE(nn.Module):
    def __init__(self):
        super(weighted_FTL_BCE, self).__init__()
        self._name = "weighted_FTL_BCE"
        weight = torch.tensor(0.01/areas[2:]) # without Hautrötung and background
        self.weights = weight.to(DEVICE)
        self.beta = 0.3
        self.alpha = 1-self.beta
        self.gamma = 1.3


    def forward(self, inputs, targets, smooth=1):

        reduce_axis = list(range(2, len(inputs.shape)))

        # True Positives, False Positives & False Negatives
        TP = torch.sum(targets * inputs, axis=reduce_axis)
        FP = torch.sum((1-targets) * inputs, axis=reduce_axis)
        FN = torch.sum(targets * (1-inputs), axis=reduce_axis)
        
        Tversky = (TP / (TP  + self.alpha*FN + self.beta*FP + smooth))

        FocalTversky = torch.mean(self.weights * (1 - Tversky)**self.gamma)
        
        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.weights * torch.mean(targets * torch.log(inputs + eps), axis=reduce_axis) \
            - torch.mean((1 - targets) * torch.log(1 - inputs + eps), axis=reduce_axis)
        BCE = BCE.mean()

        return 0.5*FocalTversky + 0.5*BCE
