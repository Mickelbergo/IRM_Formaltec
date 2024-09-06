import os
from PIL import Image

# Specify the folder containing the images
folder_path = "C:/users/comi/Desktop/Wound_segmentation_III/Data/"

# Set to store all unique RGB values
all_unique_rgb_values = set()

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Construct the full image path
        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        img = Image.open(image_path)
        
        # Convert image to RGB (if not already in RGB mode)
        img = img.convert('RGB')
        
        # Get the pixel data and update the set of unique RGB values
        pixels = list(img.getdata())
        all_unique_rgb_values.update(pixels)

# Print all unique RGB values
for rgb in all_unique_rgb_values:
    print(rgb)


### pixel valueas from 0 (background) to 210 -> 14 classes
### pixel value 160 (class 11) does not exist

