import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json


def approximate_mask_with_lines(binary_mask, epsilon_factor=0.001, min_epsilon=0.5, max_epsilon=2.0):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approximated_edges = []

    for contour in contours:
        # Calculate epsilon (maximum distance from contour to approximated contour)
        arc_length = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * (mask.size / arc_length)  # more accurate for larger wound areas
        epsilon = max(min_epsilon, min(epsilon, max_epsilon))  # maximum and minimum accuracy

        approx = cv2.approxPolyDP(contour, epsilon, True)

        edge_coordinates = [(point[0][0], point[0][1]) for point in approx]
        approximated_edges.append(edge_coordinates)

    return approximated_edges


wound_classes = ['background', 'Ungeformter_Bluterguss', 'Geformter_Bluterguss', 'Stich', 'Schnitt',
                 'Thermische_Gewalt', 'Hautabschuerfung', 'Punktfoermige_Gewalt_Schuss', 'Quetsch_Riss_Wunden',
                 'Halbscharfe_Gewalt']

min_side = 640
max_side = 1280
image_path = r"E:\ForMaLTeC\Tools\Wound_segmentation\Code\Docker_example_wound_segmentation\example_images"
mask_path = r"E:\ForMaLTeC\Tools\Wound_segmentation\Code\Docker_example_wound_segmentation\results"
annotation_path = r"E:\ForMaLTeC\Tools\Wound_segmentation\Data\Annotations"

mask_names = os.listdir(mask_path)

with open(os.path.join(annotation_path, "via_project_31May2024_22h28m.json"), "r") as via_file:
    VIA = json.load(via_file)

for n, mask_name in enumerate(mask_names[:]):
    print(n)
    mask = np.load(os.path.join(mask_path, mask_name))
    image = cv2.imread(os.path.join(image_path, mask_name[:-4]))
    height, width, cns = image.shape

    smallest_side = min(height, width)
    scale = min_side / smallest_side

    largest_side = max(height, width)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    for channel in range(mask.shape[0]):
        wound_mask = mask[channel]

        if np.sum(wound_mask) == 0:  # no wound of that class
            continue

        edge_coordinates = approximate_mask_with_lines(wound_mask, epsilon_factor=0.001, min_epsilon=0.25,
                                                       max_epsilon=2.0)

        # output_image = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2BGR)
        # for region in edge_coordinates:
        #     for i in range(len(region)):
        #         start_point = region[i]
        #         end_point = region[(i + 1) % len(region)]
        #         cv2.line(output_image, start_point, end_point, (0, 255, 0), 2)
        #
        # plt.figure(figsize=(10, 10))
        # plt.imshow(output_image)
        # plt.title("Approximated Mask with Straight Lines")
        # plt.show()

        for file in VIA["_via_img_metadata"]:
            if mask_name[:-4] == VIA["_via_img_metadata"][file]["filename"]:
                regions = []
                for edge_coordinate in edge_coordinates:
                    region = {}
                    region['region_attributes'] = {'Wound_classes_v2': wound_classes[channel]}

                    coordinates = {}
                    coordinates['name'] = 'polygon'
                    # the images were rescaled for the prediction. This rescaling has to be reversed
                    coordinates['all_points_x'] = [int(x * 1 / scale) for x, y in edge_coordinate]
                    coordinates['all_points_y'] = [int(y * 1 / scale) for x, y in edge_coordinate]

                    region['shape_attributes'] = coordinates
                    regions.append(region)

                VIA["_via_img_metadata"][file]["regions"] = regions

with open(os.path.join(annotation_path, "via_project_31May2024_22h28m_prediction.json"), "w") as f:
    json.dump(VIA, f)
