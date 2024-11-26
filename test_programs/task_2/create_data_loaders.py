import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

# Directory paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"  # Path to processed images directory

# Ensure the processed images directory exists
if not os.path.exists(processed_images_dir):
    raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

# Define a custom Dataset class
class ProcessedImagesDataset(Dataset):
    def __init__(self, images_dir):
        """
        Args:
            images_dir (str): Path to the directory containing processed images.
        """
        self.images_dir = images_dir
        
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
        
        return image, img_name  # We return the image and its filename for reference


# Create dataset object for the processed images
dataset = ProcessedImagesDataset(images_dir=processed_images_dir)

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
