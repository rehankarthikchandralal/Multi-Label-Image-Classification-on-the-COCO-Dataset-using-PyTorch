import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


image_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),           # Randomly crop and resize the image to 224x224
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.RandomRotation(30),               # Randomly rotate the image by up to 30 degrees
    transforms.ToTensor(),                       # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Function to apply transformations to all images in the directory
def apply_transformations(image_dir, transform):
    # Get all image files from the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through all images and apply transformations
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB if needed

        # Apply the transformations
        transformed_image = transform(image)
        
        # You can save or process the transformed image as needed, for example:
        # Save the transformed image
        transformed_image_path = os.path.join(image_dir, f"transformed_{image_file}")
        transformed_image.save(transformed_image_path)  # This might need to be converted back to PIL Image or used as tensor for training

# Example usage
apply_transformations(image_dir, transform)
