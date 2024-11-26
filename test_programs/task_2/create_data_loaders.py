import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

"""
This script defines a custom Dataset for loading and processing images from a specified directory, 
applies transformations including normalization, and creates DataLoader objects for training and 
validation datasets.

Functions:
- A custom `ProcessedImagesDataset` class is defined to load images, apply transformations, 
  and return images along with their filenames.
- The dataset is split into training and validation sets (80%/20%) using `train_test_split`.
- `DataLoader` objects are created for both the training and validation sets with a batch size of 16.

Modules used:
- os: For file handling
- PIL (Pillow): For opening and processing images
- torch.utils.data: For creating Dataset and DataLoader objects
- sklearn.model_selection: For splitting the dataset into training and validation sets
- torchvision.transforms: For applying image transformations (ToTensor, Normalize)
"""

# Directory paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"  # Path to processed images directory

# Ensure the processed images directory exists
if not os.path.exists(processed_images_dir):
    raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

# Define a custom Dataset class
class ProcessedImagesDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing processed images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Get all image filenames from the directory
        self.image_filenames = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        
    def __len__(self):
        # Return the number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Open the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, img_name  # Return the image and its filename for reference


# Define transformations for converting to tensor and normalizing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Create dataset object for the processed images with transformations
dataset = ProcessedImagesDataset(images_dir=processed_images_dir, transform=transform)

# Split dataset into train and validation sets (80% train, 20% validation)
train_filenames, val_filenames = train_test_split(dataset.image_filenames, test_size=0.2, random_state=42)

# Create DataLoader objects for train and validation datasets
train_dataset = torch.utils.data.Subset(dataset, [dataset.image_filenames.index(f) for f in train_filenames])
val_dataset = torch.utils.data.Subset(dataset, [dataset.image_filenames.index(f) for f in val_filenames])

# Create DataLoader objects with batch size of 16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print sample outputs to verify
for batch_idx, (images, filenames) in enumerate(train_loader):
    print(f"Batch {batch_idx+1} - Image Shape: {images.shape} - Filenames: {filenames}")
    break  # Just show the first batch as a sample
