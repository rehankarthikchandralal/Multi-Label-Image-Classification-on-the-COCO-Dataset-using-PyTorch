import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

# Directory paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"  # Path to processed images
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"  # Path to where processed images will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define a custom Dataset class for loading the images
class ImageClassificationDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        """
        Args:
            images_dir (str): Directory containing the processed images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.jpg')]
    
    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image and its corresponding label (if needed)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image


# Define transformations for data augmentation, resizing, and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Create dataset object for the processed images
dataset = ImageClassificationDataset(images_dir=processed_images_dir, transform=transform)

# Split dataset into train and validation sets (80% train, 20% validation)
train_paths, val_paths = train_test_split(dataset.image_paths, test_size=0.2, random_state=42)

# Create new dataset objects for the train and validation sets
train_dataset = ImageClassificationDataset(images_dir=processed_images_dir, transform=transform)
val_dataset = ImageClassificationDataset(images_dir=processed_images_dir, transform=transform)

# Update image paths for the training and validation sets
train_dataset.image_paths = train_paths
val_dataset.image_paths = val_paths

# Create DataLoader objects for train and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print out a few samples to verify
for batch_idx, (images) in enumerate(train_loader):
    print(f"Batch {batch_idx+1} - Image Shape: {images.shape}")
    break  # Just show the first batch as a sample
