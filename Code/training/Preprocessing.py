import os
import torch
import cv2
import kornia as K
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from scipy import stats
from augmentations import Augmentation, ValidationAugmentation 
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor
from tqdm import tqdm
import warnings

class Dataset(BaseDataset):
    def __init__(self, 
                 dir_path, 
                 image_ids, 
                 mask_ids, 
                 detection_model = None, 
                 augmentation=None, 
                 preprocessing_fn = None, 
                 target_size=(640, 640), 
                 preprocessing_config = None, 
                 train_config = None, 
                 device = None,
                 classes_to_exclude = None,
                 exclude_images_with_classes = False):
        
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.dir_path = dir_path
        self.augmentation = augmentation
        self.detection_model = detection_model
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_config = preprocessing_config
        self.train_config = train_config
        self.device = device
        self.classes_to_exclude = classes_to_exclude
        self.exclude_images_with_classes = exclude_images_with_classes


        if(self.exclude_images_with_classes and self.preprocessing_config["segmentation"] == "multiclass"):
            self.filter_images()

    def _resolve_paths_from_entry(self, entry):
        """
        Accepts:
        - str:                  "image.png"
        - (folder, image):      (".../images", "image.png")
        - (folder, image, mask) (".../images", "image_gen.png", "image.png")  # mask stripped of _gen already
        Returns (image_path, mask_path).
        For 2-tuple or str, the mask is derived by stripping '_gen' (if present) and forcing .png.
        """
        if isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                img_folder, img_name, mask_name = entry
            elif len(entry) == 2:
                img_folder, img_name = entry
                base, _ext = os.path.splitext(os.path.basename(img_name))
                if base.endswith("_gen"):
                    base = base[:-4]
                mask_name = base + ".png"
            else:
                raise TypeError(f"Unsupported entry tuple length {len(entry)} for: {entry}")
        else:
            img_folder = os.path.join(self.dir_path, "new_images_640_1280")
            img_name = entry
            base, _ext = os.path.splitext(os.path.basename(entry))
            # originals don’t have _gen, but this is safe either way
            if base.endswith("_gen"):
                base = base[:-4]
            mask_name = base + ".png"

        image_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(self.dir_path, "new_masks_640_1280", mask_name)
        return image_path, mask_path


    def filter_images(self):
        """
        Filters out images that contain any of the classes specified in classes_to_exclude.
        Updates self.image_ids and self.mask_ids in place.
        """
        filtered_image_ids = []
        filtered_mask_ids = []
        excluded_images = []
        print("Filtering images containing classes:", self.classes_to_exclude)
        for idx in tqdm(range(len(self.image_ids)), desc="Filtering images"):
            # Resolve mask from image entry (supports tuples and _gen names)
            entry = self.image_ids[idx]
            _, mask_path = self._resolve_paths_from_entry(entry)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Mask not found for image {entry}. Skipping.")
                continue  # Skip if mask cannot be read
            
            # Convert mask values to class labels
            multiclass_mask = (mask // 15).astype(np.int32) 
            
            # Check if any of the classes_to_exclude are present in the mask
            if not np.isin(self.classes_to_exclude, multiclass_mask).any():
                # If none of the excluded classes are present, keep the image
                filtered_image_ids.append(entry)
                filtered_mask_ids.append(os.path.basename(mask_path))
            else:
                # Exclude the image
                excluded_images.append(str(entry))
        
        print(f"Excluded {len(excluded_images)} images containing classes {self.classes_to_exclude}")
        self.image_ids = filtered_image_ids
        self.mask_ids = filtered_mask_ids

        # Optionally, save the list of excluded images for reference
        excluded_images_path = os.path.join(self.dir_path, "excluded_images.txt")
        with open(excluded_images_path, 'w') as f:
            for img in excluded_images:
                f.write(f"{img}\n")
        print(f"List of excluded images saved to {excluded_images_path}")

    def detect_and_crop(self, image, mask, margin=200): #freely change the margin
        """Detect regions using YOLO, add a margin around them, and crop image and mask."""
        if self.detection_model:
            results = self.detection_model.predict(source=np.array(image), device=self.device, save=False, verbose=False)
            detections = results[0].boxes

            if len(detections) > 0:
                # Get the bounding box with the highest confidence
                x1, y1, x2, y2 = map(int, detections.xyxy[0].cpu().numpy())

                # Add margin
                left_margin = np.random.randint(0, margin)
                top_margin = np.random.randint(0, margin)
                right_margin = np.random.randint(0, margin)
                bottom_margin = np.random.randint(0, margin)
                x1 = max(0, x1 - left_margin)
                y1 = max(0, y1 - top_margin)
                x2 = min(image.width, x2 + right_margin)
                y2 = min(image.height, y2 + bottom_margin)

                # Crop and resize image and mask
                cropped_image = image.crop((x1, y1, x2, y2)).resize(self.target_size)
                cropped_mask = mask.crop((x1, y1, x2, y2)).resize(self.target_size, Image.NEAREST)
            else:
                # If no detection, fallback to resizing the entire image
                #print("No YOLO detections found. Using full image resize.")
                return self.resize(image, mask)
        else:
            # If no detection model is provided, resize the full image and mask
            return self.resize(image, mask)

        return cropped_image, cropped_mask


    def resize(self, image, mask):
        """Resize image and mask to the target size."""
        resized_image = image.resize(self.target_size)
        resized_mask = mask.resize(self.target_size, Image.NEAREST)
        return resized_image, resized_mask

    def extract_background(self, image, mask, patch_size=(224, 224)):
        """Extract a patch of the background (class 0) from the image."""
        # Convert mask to a NumPy array for easier processing
        mask_array = np.array(mask)

        # Find all background pixels (label 0)
        background_coords = np.argwhere(mask_array == 0)

        if len(background_coords) == 0:
            # If no background is found, fall back to center crop
            print("No background found in mask. Performing center crop.")
            return self.resize(image, mask)

        # Randomly sample a background coordinate
        random_index = np.random.choice(len(background_coords)) 
        center_y, center_x = background_coords[random_index]

        # Define patch boundaries
        half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(image.width, center_x + half_w)
        y2 = min(image.height, center_y + half_h)

        # Crop the image and mask around the sampled background patch
        cropped_image = image.crop((x1, y1, x2, y2)).resize(self.target_size)
        cropped_mask = mask.crop((x1, y1, x2, y2)).resize(self.target_size, Image.NEAREST)

        return cropped_image, cropped_mask

    def __getitem__(self, ind):
        # Load image and mask
        # image_path = os.path.sep.join([self.dir_path, "new_images_640_1280", self.image_ids[ind]])
        # mask_path = os.path.sep.join([self.dir_path, "new_masks_640_1280", self.mask_ids[ind]])
        image_path, mask_path = self._resolve_paths_from_entry(self.image_ids[ind])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            if mask is None:
                warnings.warn(
                    f"[Dataset] Mask not found for image entry {self.image_ids[ind]} (looked for {mask_path}). "
                    f"If this is a generated image ('*_gen.png'), ensure a matching mask exists without the '_gen' suffix.",
                    RuntimeWarning
                )
            raise ValueError("mask is none")
        

        # Convert mask to binary segmentation mask
        binary_mask = (mask > 0).astype(np.uint8)
        # Convert mask values to class labels

        multiclass_mask = (mask // 15)  # Assuming mask values are wound_class * 15

        multiclass_mask[np.isin(multiclass_mask, [11, 12, 13, 14])] = 6 #this gets rid of classes 11-14. #remove this (and recalculate weights) to revert


        # Filter out background (class 0) and get non-background classes
        non_background_pixels = multiclass_mask[multiclass_mask != 0]
        dominant_class = 0



        image = Image.fromarray(image)
        binary_mask = Image.fromarray(binary_mask)
        multiclass_mask = Image.fromarray(multiclass_mask)


        #change this if you want, with probability p[0], a cropped wound image is used and with p[1] the original image is used
        mode = np.random.choice(["yolo", "resize"], p = [0.4,0.6])

        if self.augmentation == "train":
            
            if self.preprocessing_config["segmentation"] == "binary":
                if mode == "yolo":
                    image, binary_mask= self.detect_and_crop(image, binary_mask)
                elif mode == "resize":
                    image, binary_mask= self.resize(image, binary_mask)
                elif mode == "background":
                    image, binary_mask= self.extract_background(image, binary_mask)
                # else:
                #     raise ValueError(f"Unknown mode: {mode}")

            elif self.preprocessing_config["segmentation"] == 'multiclass':
                if mode == "yolo":
                    image, multiclass_mask = self.detect_and_crop(image, multiclass_mask)
                elif mode == "resize":
                    image, multiclass_mask = self.resize(image, multiclass_mask)
                elif mode == "background":
                    image, multiclass_mask = self.extract_background(image, multiclass_mask)
                # else:
                #     raise ValueError(f"Unknown mode: {mode}")
            
            else:
                raise ValueError(f'segmentation must either be "binary" or "multiclass"')
            
       
        # Convert to tensor 
        #note that Kornia permutes the images directly, no need to manually permute
        image = np.array(image)
        binary_mask = np.array(binary_mask)
        multiclass_mask = np.array(multiclass_mask)

        image = K.image_to_tensor(image).float().to(self.device)
        binary_mask = K.image_to_tensor(binary_mask).long().to(self.device)  # Binary mask for segmentation
        multiclass_mask = K.image_to_tensor(multiclass_mask).long().to(self.device)  # Multiclass segmentation

        #self.visualize_sample(image, multiclass_mask, binary_mask)

        # Apply augmentations
        if self.augmentation == 'train':
            if self.preprocessing_config["segmentation"] == "binary":
                image, binary_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, binary_mask)
            else: image, multiclass_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, multiclass_mask)
        
        if self.augmentation == 'validation':
            if self.preprocessing_config["segmentation"] == "binary":
                image, binary_mask = ValidationAugmentation(self.target_size, self.preprocessing_fn).augment(image, binary_mask)
            else: image, multiclass_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, multiclass_mask)

        return image, binary_mask, multiclass_mask, 0
    

    def __len__(self):
        return len(self.image_ids)


    def visualize_sample(self, image, multiclass_mask, binary_mask):
        # Convert tensors back to numpy
        image_np = K.tensor_to_image(image.cpu().long())
        binary_mask_np = binary_mask.cpu().numpy()[0]  # Assuming the mask has shape [1, H, W]
        multiclass_mask_np = multiclass_mask.cpu().numpy()[0]  # Assuming the mask has shape [1, H, W]

        # Get the unique class labels in the multiclass mask
        unique_classes = np.unique(multiclass_mask_np)

        # Plot the image, binary mask, and multiclass mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title("Image")

        axes[1].imshow(binary_mask_np, cmap="gray")
        axes[1].set_title("Binary Mask")

        axes[2].imshow(multiclass_mask_np, cmap="nipy_spectral")
        axes[2].set_title(f"Multiclass Mask - Classes: {unique_classes}")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()



class TransformerDataset(BaseDataset):
    """
    A dataset class for semantic segmentation using a Hugging Face segmentation processor.
    This class delegates standard preprocessing to the processor and retains only
    custom preprocessing steps like detection-based cropping or background extraction.
    """

    def __init__(
        self,
        dir_path,
        image_ids,
        mask_ids,
        augmentation=None, 
        preprocessing_fn=None,
        preprocessing_config=None,
        train_config=None,
        device=None,
        detection_model=None,
        margin=200,
        use_background_extraction=False,
        processor_name="nvidia/segformer-b0-finetuned-ade-512-512"  # Change as needed
    ):
        """
        Args:
            dir_path (str): Root directory of the dataset.
            image_ids (List[str]): List of image filenames.
            mask_ids (List[str]): List of mask filenames.
            augmentation (str): 'train', 'validation', or None — used to control augmentations.
            preprocessing_fn (callable): A callable used for image normalization, etc.
            preprocessing_config (dict): Config dict (e.g., "binary" or "multiclass").
            train_config (dict): Additional training config if needed.
            device (torch.device): The device to put the tensors on (optional).
            detection_model: YOLO model if you still want to do detect-and-crop (optional).
            margin (int): The max margin around detections if using YOLO.
            use_background_extraction (bool): Whether to randomly crop a background patch.
            processor_name (str): HF processor checkpoint name for segmentation.
        """

        self.dir_path = dir_path
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_config = preprocessing_config
        self.train_config = train_config
        self.device = device
        self.detection_model = detection_model
        self.margin = margin
        self.use_background_extraction = use_background_extraction

        # Initialize the Hugging Face image processor
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        # If you have a specific processor, e.g., SegformerImageProcessor, import and use it instead:
        # from transformers import SegformerImageProcessor
        # self.processor = SegformerImageProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.image_ids)

    def _resolve_paths_from_entry(self, entry):
        """
        Accepts:
        - str:                  "image.png"
        - (folder, image):      (".../images", "image.png")
        - (folder, image, mask) (".../images", "image_gen.png", "image.png")
        Returns (image_path, mask_path). For 2-tuple or str, derive mask by stripping '_gen' and using .png.
        """
        if isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                img_folder, img_name, mask_name = entry
            elif len(entry) == 2:
                img_folder, img_name = entry
                base, _ext = os.path.splitext(os.path.basename(img_name))
                if base.endswith("_gen"):
                    base = base[:-4]
                mask_name = base + ".png"
            else:
                raise TypeError(f"Unsupported entry tuple length {len(entry)} for: {entry}")
        else:
            img_folder = os.path.join(self.dir_path, "new_images_640_1280")
            img_name = entry
            base, _ext = os.path.splitext(os.path.basename(entry))
            if base.endswith("_gen"):
                base = base[:-4]
            mask_name = base + ".png"

        image_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(self.dir_path, "new_masks_640_1280", mask_name)
        return image_path, mask_path


    def detect_and_crop(self, image, mask):
        """
        (Optional) YOLO-based detection, then crop & resize. Returns PIL images.
        """
        if self.detection_model:
            results = self.detection_model.predict(
                source=np.array(image), device=self.device, save=False, verbose=False
            )
            detections = results[0].boxes
            if len(detections) > 0:
                # Take the first detection (or the one with highest confidence)
                x1, y1, x2, y2 = map(int, detections.xyxy[0].cpu().numpy())
                # Random margins
                left_margin = np.random.randint(0, self.margin)
                top_margin = np.random.randint(0, self.margin)
                right_margin = np.random.randint(0, self.margin)
                bottom_margin = np.random.randint(0, self.margin)
                x1 = max(0, x1 - left_margin)
                y1 = max(0, y1 - top_margin)
                x2 = min(image.width, x2 + right_margin)
                y2 = min(image.height, y2 + bottom_margin)

                # Crop
                image = image.crop((x1, y1, x2, y2))
                mask = mask.crop((x1, y1, x2, y2))

        # If no detection or detection_model is None, fallback to resize entire image
        return image, mask

    def extract_background(self, image, mask, patch_size=(224, 224)):
        """
        Randomly extract a patch of background (class 0) from the image, then resize.
        """
        mask_array = np.array(mask)
        background_coords = np.argwhere(mask_array == 0)
        if len(background_coords) == 0:
            # Fallback to resizing if no background found
            return self.resize(image, mask)

        # Randomly choose one background pixel as center
        random_index = np.random.choice(len(background_coords))
        center_y, center_x = background_coords[random_index]

        half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(image.width, center_x + half_w)
        y2 = min(image.height, center_y + half_h)

        # Crop and resize
        image = image.crop((x1, y1, x2, y2))
        mask = mask.crop((x1, y1, x2, y2))
        return image, mask

    def resize(self, image, mask):
        """
        Resize to target size. Returns PIL images.
        """
        # Note: Resizing is handled by the processor, so this can be optional.
        # If you decide to keep it for specific reasons, ensure it aligns with processor settings.
        image = image.resize(self.processor.size["height"], self.processor.size["width"])
        mask = mask.resize(self.processor.size["height"], self.processor.size["width"], Image.NEAREST)
        return image, mask

    def preprocess_masks(self, mask_np):
        """
        Process the mask according to the preprocessing_config.
        For example, convert to binary or multiclass masks.
        """
        if self.preprocessing_config:
            if self.preprocessing_config.get("segmentation") == "binary":
                # Binary mask: 0 for background, 1 for foreground
                mask_np = (mask_np > 0).astype(np.uint8)
            elif self.preprocessing_config.get("segmentation") == "multiclass":
                # Multiclass mask: Assuming mask values are wound_class * 15
                mask_np = mask_np // 15
                # Remove or merge specific classes if needed
                mask_np[np.isin(mask_np, [11, 12, 13, 14])] = 6  # Example: merging classes 11-14 into class 6
            else:
                raise ValueError('segmentation must either be "binary" or "multiclass"')
        return mask_np

    def __getitem__(self, idx):
        # 1) Load image & mask paths
        # image_path = os.path.join(self.dir_path, "new_images_640_1280", self.image_ids[idx])
        # mask_path = os.path.join(self.dir_path, "new_masks_640_1280", self.mask_ids[idx])
        image_path, mask_path = self._resolve_paths_from_entry(self.image_ids[idx])

        # 2) Load image & mask using OpenCV
        image_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image_cv2 is None or mask_cv2 is None:
            raise ValueError(f"Could not read image or mask: {image_path}, {mask_path}")

        # Convert BGR to RGB and to PIL Image
        image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask_cv2)

        # 3) Apply custom preprocessing: detect and crop or extract background
        if self.augmentation == "train":
            if self.use_background_extraction:
                # Example probabilities: YOLO=0.2, resize=0.7, background=0.1
                mode = np.random.choice(["yolo", "resize", "background"], p=[0.2, 0.7, 0.1])
            else:
                # Example probabilities: YOLO=0.4, resize=0.6
                mode = np.random.choice(["yolo", "resize"], p=[0.2, 0.8])

            if mode == "yolo" and self.detection_model:
                image, mask = self.detect_and_crop(image, mask)
            elif mode == "background" and self.use_background_extraction:
                image, mask = self.extract_background(image, mask)
            else:
                image, mask = self.resize(image, mask)
        else:
            # For validation or testing, only resize
            image, mask = self.resize(image, mask)

        # 4) Convert mask to numpy array and preprocess
        mask_np = np.array(mask)
        mask_np = self.preprocess_masks(mask_np)

        # 5) Use the Hugging Face processor to handle image and mask preprocessing
        # The processor will handle resizing, normalization, etc.
        # Ensure that masks are in the correct format (single-channel with class IDs)
        encoding = self.processor(
            images=image,
            masks=mask_np,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze()  # shape: (C, H, W)
        labels = encoding["labels"].squeeze()              # shape: (H, W)

        # 6) Apply additional augmentations if necessary
        # If augmentations.py expects tensors, apply them after processor
        if self.augmentation == "train":
            if self.preprocessing_config and self.preprocessing_config.get("segmentation") == "binary":
                pixel_values, labels = Augmentation(
                    self.processor.size, self.preprocessing_fn
                ).augment(pixel_values, labels)
            else:
                pixel_values, labels = Augmentation(
                    self.processor.size, self.preprocessing_fn
                ).augment(pixel_values, labels)
        elif self.augmentation == "validation":
            if self.preprocessing_config and self.preprocessing_config.get("segmentation") == "binary":
                pixel_values, labels = ValidationAugmentation(
                    self.processor.size, self.preprocessing_fn
                ).augment(pixel_values, labels)
            else:
                pixel_values, labels = ValidationAugmentation(
                    self.processor.size, self.preprocessing_fn
                ).augment(pixel_values, labels)

        # 7) Move tensors to the specified device, if any
        if self.device:
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)

        # 8) Return the processed tensors
        # You can also return additional information if needed
        return pixel_values, labels, 0  # The '0' can be a placeholder for other data (e.g., image ID)

    def visualize_sample(self, image, label):
        """
        Optional: Visualize the image and its corresponding label/mask.
        Useful for debugging and ensuring preprocessing is correct.
        """
        import matplotlib.pyplot as plt

        image_np = K.tensor_to_image(image.cpu().long())
        label_np = label.cpu().numpy()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(label_np, cmap="nipy_spectral")
        plt.title("Mask")
        plt.axis("off")

        plt.show()
