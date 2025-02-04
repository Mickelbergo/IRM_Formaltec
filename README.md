# Wound Segmentation

## Description

The purpose of this project is to automatically detect, classify and segment different types of forensic wounds. For that I used different kind of Convolutational neural networks and transformer networks.

## Usage

### Preprocessing
Before using the main file, first preprocessing needs to be performed by executing preprocessing.py, weights.py and yolo.py.

This resizes and saves new images, calculates weights for the differnt classes (for multiclass segmentation) and trains a yolo model to detect wounds based on bounding boxes for further augmentations afterwards


### In config files

#### Change the path

My current paths:

my PC: 
"C:/users/comi/Desktop/Wound_segmentation_III/Data" 
"C:/Users/comi/Desktop/Wound_Segmentation_III/GIT/IRM_Formaltec/New_Code/training"

train PC: 
"E:/ForMalTeC/Wound_segmentation_III/Data" 
"E:/ForMaLTeC/Wound_segmentation_III/GIT/IRM_Formaltec/New_Code/training"

train PC2: 
"C:/users/comi/Desktop/Wound_Segmentation_III/Data" 
"C:/Users/comi/Desktop/Wound_Segmentation_III/IRM_Formaltec/New_Code/training"


#### To change from binary to multiclass:
-Change segmentation classes from 11 to 2 in training_config
-Change segmentation to "binary" from "multiclass" in preprocessing_config

-The other parameters can be tuned or turned on/off

#### To change to transformers:
To use the transformers library, simply change the encoder = "transformer" (this uses a swin_b transformer at the moment, this can be changed in the model.py)

Building a self-made transformer is still in progress (in preprocessing.py TransformerDataset and in main_transformer.py)


## Current classes

0 = background
1 = dermatorrhagia / ungeformter bluterguss
2 = hematoma /geformter bluterguss
3 = stab / stich
4 = cut / schnitt
5 = thermal / thermische gewalt
6 = skin abrasion /hautabschürfung
7 = puncture-gun shot / punktförmige-gewalt-schuss
8 = contused-lacarated / quetsch-riss Wunden (Platzwunden)
9 = semisharp force / Halbscharfe Gewalt
10 = lacerations / risswunden

### Removed classes

I got rid of these classes, they were all put to class 6
11 = non-existent
12 = ungeformter bluterguss + hautabschürfung
13 = geformter bluterguss + hautabschürfung
14 = thermische gewalt + hautabschürfung -->

### Merging classes

Classes 3, 7 and 9 do not occur frequently, so we need an option to turn them off
This can be configured in preprocessing.py but the implementation is still work in progress

## Other things that can be changed

yolo version

the margin used on yolo pictures to crop them (Preprocessing.py -> detect_and_crop(margin))

-the probabilities of using mode = ["yolo", "resize", "background"] (preprocessing.py __get_item__)

-the way the weights for multiclass segmentation get calculated (main_gridsearch -> weight_ranges)

-the augmentations

-the model itself (Unet/Unetplusplus/Deeplab/Huggingface) (model.py)

-learning rate

-optimizer

-weights


## Current best models

### Binary

 best_model_v1.4_epoch40_encoder_se_resnext101_32x4d_seg_binary_lambda1.0_optadamw_lr0.0001_dice+ce_wr50_200_samplerTrue_iou0.7582_f10.8403.pth

-lr scheduler gamma = 0.999

-adamW

-num workers 10, batch size 12

### Multiclass

-swin_b transformer

-lr: 0.0001

-lr scheduler gamma = 0.999

-adamW

-num workers 10, batch size 16
