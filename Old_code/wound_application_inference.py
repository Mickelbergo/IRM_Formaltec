'''
Code for Wound application inference
'''
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path_to_images = r"E:\ForMaLTeC\Tools\Wound_segmentation\Code\Docker_example_wound_segmentation\example_images"

# path_to_model = r"C:\Users\adobay\Documents\ForMaLTeC\Tools\Wound_segmentation\Data\runs\paper\640_FPN_mit_b4_sigmoid_fulldata_low_FN_2024_01_08_19\640_FPN_mit_b4_sigmoid_fulldata_low_FN_best_model_fold_0.pth"

base = r"E:\ForMaLTeC\Tools\Wound_segmentation\Data\runs\paper"
path_to_model = os.path.join(base, "640_FPN_mit_b4_sigmoid_fulldata_low_FN_2024_01_08_19", "640_FPN_mit_b4_sigmoid_fulldata_low_FN_best_model_fold_0.pth")
# path_to_model = os.path.join(base, "640_FPN_mit_b4_sigmoid_no_classes_2024_01_12_17", "640_FPN_mit_b4_sigmoid_no_classes_75_epoch_model_fold_0.pth")
# path_to_model = os.path.join(base, "640_FPN_mit_b4_sigmoid_no_classes_2024_01_22_18","640_FPN_mit_b4_sigmoid_no_classes_75_epoch_model_fold_0.pth")
# path_to_model = os.path.join(base, "640_FPN_mit_b4_sigmoid_no_classes_2024_01_22_18","640_FPN_mit_b4_sigmoid_no_classes_last_epoch_model_fold_0.pth")
# path_to_model = os.path.join(base, "640_FPN_mit_b4_sigmoid_no_classes_label_smoothing_2024_02_02_17", "640_FPN_mit_b4_sigmoid_no_classes_label_smoothing_last_epoch_model_fold_0.pth")
# path_to_model = os.path.join(base, "640_FPN_mit_b4_softmax2d_no_classes_label_smoothing_2024_02_05_15", "640_FPN_mit_b4_softmax2d_no_classes_label_smoothing_75_epoch_model_fold_0.pth")
path_to_output_folder = r"C:\Users\adobay\Documents\ForMaLTeC\Tools\Wound_segmentation\Code\Docker_example_wound_segmentation\results" 

image_paths = os.listdir(path_to_images)

min_side = 512 # beim Training wird alles auf 512x512 geschnitten
max_side = 1024

mean = torch.Tensor([0.55540512, 0.46654144, 0.42994756]) # mean calculated for our wound dataset
std = torch.Tensor([0.21014232, 0.21117639, 0.22304179]) 
wound_classes = ['Ungeformter_Bluterguss', 'Geformter_Bluterguss', 'Stich', 'Schnitt', 'Thermische_Gewalt', 'Hautabschuerfung', \
                 'Punktfoermige_Gewalt_Schuss', 'Quetsch_Riss_Wunden', 'Halbscharfe_Gewalt']
    
if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

model = torch.load(path_to_model).to(DEVICE)
cmap = plt.colormaps.get_cmap("tab10")

for image_path in image_paths[:]:
    image: np.ndarray = cv2.imread(os.path.join(path_to_images, image_path), cv2.IMREAD_COLOR)
    
    try:
        height, width, cns = image.shape
    except:
        print(image_path)
    
    smallest_side = min(height, width)
    scale = min_side/smallest_side
    
    largest_side = max(height, width)
    
    if largest_side * scale > max_side:
        scale = max_side / largest_side
        
    small_rows = int(round(width*scale))
    small_cols = int(round(height*scale))
    
    image_small = cv2.resize(image, (small_rows , small_cols))
    
    rows, cols, cns = image_small.shape
    
    pad_w = (32-small_rows%32)%32
    pad_h = (32-small_cols%32)%32
    
    new_image = np.zeros(((small_cols + pad_h), (small_rows + pad_w), cns)).astype(np.uint8)
    
    new_image[:rows, :cols, :] = image_small.astype(np.uint8)
    
    plt.figure(figsize=(int(rows/50), int(cols/50)))
    plt.imshow(new_image[:,:,[2,1,0]])
    image = new_image.copy()
    # plt.show()
    
    new_image: torch.Tensor = torch.from_numpy(new_image).permute((2,0,1))
    new_image = new_image.type(torch.float32).to(DEVICE)

    new_image = new_image / 255.0
    
    new_image[0] -= mean[0]
    new_image[1] -= mean[1]
    new_image[2] -= mean[2]
    
    new_image[0] /= std[0]
    new_image[1] /= std[1]
    new_image[2] /= std[2]
    
    with torch.no_grad():
        model.eval()
        prediction = model.forward(new_image.unsqueeze(0))[0]
        pred = prediction.cpu()
        
    # prediction[prediction>0.5] = 1
    # prediction[prediction<=0.5] = 0
    
    pred[1:] = (pred[1:]-pred[1:].min())/(pred[1:].max()-pred[1:].min())
    
    pred_max = torch.argmax(pred, axis=0)
    prediction = torch.permute(F.one_hot(pred_max, num_classes = 11), (2,0,1))
    
    
    prediction = prediction.numpy().astype(np.uint8)
    
    # for i in range(1, 20):
    #     if np.sum(prediction[1:]) == 0:
    #         print("no wound")
    #         pred[0] -= (0.03)
    #         pred_max = torch.argmax(pred, axis=0)
    #         print(torch.unique(pred_max))
    #         prediction = torch.permute(F.one_hot(pred_max, num_classes = 11), (2,0,1))   
    #         prediction = prediction.numpy().astype(np.uint8)
    #     else:
    #         break

    # Postprocessing
    class_masks = []
    # Remove small wound areas
    for class_index in range(1, prediction.shape[0]):
        class_mask = prediction[class_index]
    
        n_components, labels, stats, centers = cv2.connectedComponentsWithStats(class_mask, connectivity=8) # diagonal pixels count as neighbours
        for label in range(1, stats.shape[0]):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 50:
                class_mask[labels == label] = 0

        kernel = np.ones((20,20), np.uint8)
        class_masks.append(cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel))
    
    
    for channel in range(1, prediction.shape[0]):
        mask = prediction[channel,:,:]
        mask = class_masks[channel-1]
        mask_color = cmap(channel)
        plt.contour(mask, colors=[mask_color], alpha=0.7)
        
    # plt.imshow(np.sum(prediction[1:], axis=0))
    legend_elements=[Patch(facecolor=cmap(i+1), label = wound_classes[i]) for i in range(len(wound_classes))]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.axis("off")
    plt.show()
    
    plt.figure(figsize=(int(rows/50), int(cols/50)))
    plt.imshow(image[:,:,[2,1,0]])
    
    heat = torch.sum(pred[1:], 0)
    norm_heat = (heat-heat.min())/(heat.max()-heat.min())
    rgba_heatmap = cm.jet(norm_heat)
    rgba_heatmap[:,:,3] = norm_heat
    plt.imshow(rgba_heatmap, alpha = 0.5)
    plt.axis("off")
    plt.show()

    # prediction_list = prediction.tolist()
    
    # if not os.path.exists(path_to_output_folder):
    #     os.makedirs(path_to_output_folder)
    
    # with open(os.path.join(path_to_output_folder, f"{image_path[:-4]}.json"), "w") as json_file:
    #     json.dump(prediction_list, json_file)
