import os
import cv2
import json 
def flip_and_resize_images_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, target_size=(640, 1280)):
    """
    Flip and resize all images and masks to the target size (640x1280), minimizing resizing.
    
    Args:
    - image_dir (str): Directory containing the original images.
    - mask_dir (str): Directory containing the original masks.
    - output_image_dir (str): Directory to save resized images.
    - output_mask_dir (str): Directory to save resized masks.
    - target_size (tuple): Target size (width, height) for resizing (default is 640x1280).
    
    Returns:
    None
    """
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get list of image and mask filenames
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    # Ensure the number of images and masks match
    assert len(image_filenames) == len(mask_filenames), "Mismatch between number of images and masks."
    
    for img_filename, mask_filename in zip(image_filenames, mask_filenames):
        # Load the image and mask
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Failed to load image or mask: {img_filename}, {mask_filename}")
            continue
        
        # Get image dimensions
        h, w = image.shape[:2]

        # If the image is wider than taller (e.g., 1280x640), flip it to be taller than wider
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            h, w = image.shape[:2]  # Update dimensions after flip

        # Resize the image and mask to the target size (640x1280)
        if (h, w) != target_size:
            resized_image = cv2.resize(image, target_size)
            resized_mask = cv2.resize(mask, target_size , interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for masks
        else:
            resized_image, resized_mask = image, mask  # If already the correct size, no resizing

        # Save the resized image and mask to the output directories
        output_image_path = os.path.join(output_image_dir, img_filename)
        output_mask_path = os.path.join(output_mask_dir, mask_filename)
        
        cv2.imwrite(output_image_path, resized_image)
        cv2.imwrite(output_mask_path, resized_mask)

        print(f"Processed and saved: {img_filename}, {mask_filename}")


with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)
path = train_config["path"]


image_dir =  os.path.join(path, "Images_640_1280")
mask_dir = os.path.join(path, "Masks_640_1280")
output_image_dir =  os.path.join(path, "new_images_640_1280")
output_mask_dir =  os.path.join(path, "new_masks_640_1280")

flip_and_resize_images_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, target_size=(640, 1280))
