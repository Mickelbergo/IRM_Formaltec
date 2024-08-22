import cv2

def resize_and_pad(image, mask, target_size=(640, 640)):
    """
    Resize the image and mask to maintain aspect ratio and then pad to the target size.
    
    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The corresponding mask.
        target_size (tuple): The target size (height, width) to resize and pad the image and mask to.
        
    Returns:
        padded_image (np.ndarray): The resized and padded image.
        padded_mask (np.ndarray): The resized and padded mask.
    """
    target_height, target_width = target_size
    
    # Get current dimensions
    height, width = image.shape[:2]
    
    # Resize the image and mask while keeping the aspect ratio
    if width > height:
        scale = target_width / width
        new_width = target_width
        new_height = int(height * scale)
    else:
        scale = target_height / height
        new_height = target_height
        new_width = int(width * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Pad the resized image and mask to the target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_mask = cv2.copyMakeBorder(resized_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return padded_image, padded_mask

# You can add more augmentation functions below as needed.
