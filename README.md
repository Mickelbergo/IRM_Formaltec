#my PC: "C:/users/comi/Desktop/Wound_segmentation_III/Data" "C:/Users/comi/Desktop/Wound_Segmentation_III/GIT/IRM_Formaltec/New_Code/training"
#train PC: "E:/ForMalTeC/Wound_segmentation_III/Data" "E:/ForMaLTeC/Wound_segmentation_III/GIT/IRM_Formaltec/New_Code/training"
#train PC2: "C/users/comi/Desktop/Wound_segmentation_III/Data" "C:/Users/comi/Desktop/Wound_Segmentation_III/IRM_Formaltec/New_Code/training"


to change to binary segmentation, change: 
activation: softmax -> sigmoid
loss function: cross_entropy -> bce
segmentation: multiclass -> binary


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

To effectively use a transformer model, we first perform object detection to identify the regions of interest (the wounds), and then we can crop the image based on these ROIs.
We then segment the image using a transformer
