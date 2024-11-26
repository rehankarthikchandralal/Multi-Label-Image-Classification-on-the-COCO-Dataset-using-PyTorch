import os
from PIL import Image
from torchvision import transforms
import torch

"""
This script processes images in the COCO dataset for image classification tasks. It includes functions to:

- Resize images to a fixed size of 224x224 pixels.
- Normalize images based on the ImageNet mean and standard deviation.
- Apply data augmentation techniques such as random cropping, flipping, and rotation.
- Process images by applying all the above transformations sequentially.

The script processes all `.jpg` images from a specified input directory and saves the transformed images to an output directory.

Modules used:
- os: For file handling
- PIL (Pillow): For image opening, processing, and saving
- torchvision.transforms: For various image transformations
- torch: For tensor handling (in the normalization function)
"""

# Directory paths
input_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"  # Path to images directory
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"      # Path where processed images will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to Resize Images to 224x224 pixels
def resize_image(image, size=(224, 224)):
    """
    Resize the image to a specified size.
    
    Args:
        image (PIL Image): Input image to be resized.
        size (tuple): Desired size for the image (default is 224x224).
    
    Returns:
        PIL Image: Resized image.
    """
    transform = transforms.Resize(size)
    return transform(image)


# Function to Normalize Images
def normalize_image(image):
    """
    Normalize the image based on the mean and std for the ImageNet dataset.
    
    Args:
        image (PIL Image or Tensor): Input image to be normalized.
    
    Returns:
        Tensor: Normalized image.
    """
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Convert the image to a tensor
    tensor_image = transforms.ToTensor()(image)
    return transform(tensor_image)


# Function for Data Augmentation (Random Crop, Flip, Rotation)
def augment_image(image):
    """
    Apply data augmentation techniques like random crop, random horizontal flip, and random rotation.
    
    Args:
        image (PIL Image): Input image for augmentation.
    
    Returns:
        PIL Image: Augmented image.
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random crop to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(30)       # Random rotation between -30 and 30 degrees
    ])
    return transform(image)


# Function to Apply All Transforms Together
def process_image(image):
    """
    Apply all transformations (resize, normalize, and augment) on the image.
    
    Args:
        image (PIL Image): Input image to be processed.
    
    Returns:
        Tensor: Processed image.
    """
    # Apply augmentation first (random crop, flip, rotate)
    augmented_image = augment_image(image)
    
    # Resize the image to 224x224
    resized_image = resize_image(augmented_image)
    
    # Normalize the image
    normalized_image = normalize_image(resized_image)
    
    return normalized_image

# Process all images in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    
    # Check if the file is a .jpg image
    _, extension = os.path.splitext(image_name)
    if extension.lower() == '.jpg':  # Check if the extension is exactly '.jpg'
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Test resize_image
            # resized_image = resize_image(image)
            # resized_image_pil = transforms.ToPILImage()(transforms.ToTensor()(resized_image))
            # resized_image_pil.save(os.path.join(output_dir, f"resized_{image_name}"))
            # print(f"Resized and saved: {image_name}")
            
            #  Test normalize_image
            # normalized_image = normalize_image(image)
            # normalized_image_pil = transforms.ToPILImage()(normalized_image)
            # normalized_image_pil.save(os.path.join(output_dir, f"normalized_{image_name}"))
            # print(f"Normalized and saved: {image_name}")
            
            #  Test augment_image
            # augmented_image = augment_image(image)
            # augmented_image_pil = transforms.ToPILImage()(transforms.ToTensor()(augmented_image))
            # augmented_image_pil.save(os.path.join(output_dir, f"augmented_{image_name}"))
            # print(f"Augmented and saved: {image_name}")
            
            #  Test all transformations together (process_image)
            # processed_image = process_image(image)
            # processed_image_pil = transforms.ToPILImage()(processed_image)
            # processed_image_pil.save(os.path.join(output_dir, f"processed_{image_name}"))
            # print(f"Processed and saved: {image_name}")
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
