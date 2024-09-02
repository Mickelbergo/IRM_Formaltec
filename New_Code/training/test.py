

import os
import cv2

def find_problematic_files(directory):
    problematic_files = []
    for filename in os.listdir(directory):
        try:
            filepath = os.path.join(directory, filename)
            # Attempt to read the image file using cv2
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("cv2 could not read the file, possibly not an image")
        except Exception as e:
            print(f"Problem with file: {filename} - {str(e)}")
            problematic_files.append(filename)
    return problematic_files

image_directory =  "E:/ForMalTeC/Wound_segmentation_III/Data/Images_640_1280"
problematic_files = find_problematic_files(image_directory)

if problematic_files:
    print("Found problematic files:")
    for file in problematic_files:
        print(file)
else:
    print("No problematic files found.")
