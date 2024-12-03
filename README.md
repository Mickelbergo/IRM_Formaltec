#my PC: "C:/users/comi/Desktop/Wound_segmentation_III/Data" "C:/Users/comi/Desktop/Wound_Segmentation_III/GIT/IRM_Formaltec/New_Code/training"
#train PC: "E:/ForMalTeC/Wound_segmentation_III/Data" "E:/ForMaLTeC/Wound_segmentation_III/GIT/IRM_Formaltec/New_Code/training"
#train PC2: "C:/users/comi/Desktop/Wound_Segmentation_III/Data" "C:/Users/comi/Desktop/Wound_Segmentation_III/IRM_Formaltec/New_Code/training"


to change to binary segmentation, change: 
activation: softmax -> sigmoid
segmentation: multiclass -> binary
segmentation_classes: 15 -> 2

to change to transformers:
encoder = 'transformers' (uses a swin something at the moment, change)


#0 = background
#1 = dermatorrhagia / ungeformter bluterguss
#2 = hematoma /geformter bluterguss
#3 = stab / stich
#4 = cut / schnitt
#5 = thermal / thermische gewalt
#6 = skin abrasion /hautabschürfung
#7 = puncture-gun shot / punktförmige-gewalt-schuss
#8 = contused-lacarated / quetsch-riss Wunden (Platzwunden)
#9 = semisharp force / Halbscharfe Gewalt
#10 = lacerations / risswunden
#11 = non-existent
#12 = ungeformter bluterguss + hautabschürfung
#13 = geformter bluterguss + hautabschürfung
#14 = thermische gewalt + hautabschürfung

Things that can be changed (apart from the configuration files):
-yolo version
-the margin used on yolo pictures to crop them
-the probabilities of using mode = ["yolo", "resize", "background"] (preprocessing)
-the way the weights for multiclass segmentation get calculated
-the augmentations
-the model itself (Unet/Unetplusplus/Deeplab/Huggingface)
-learning rate
-optimizer
-weights
-


good paramaters for swin_v2b (binary, stages 1-4):
-weights: [1,60]
-lr: 0.0001
-adamW
-