import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
import json
from tqdm import tqdm
import numpy as np

# Define paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"
json_file = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/instances_train.json"

# Check if directories and files exist
if not os.path.exists(processed_images_dir):
    raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON annotations file not found: {json_file}")

print("Loaded configuration successfully.")

class ProcessedImagesDataset(Dataset):
    def __init__(self, images_dir, json_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        print(f"Loading annotations from {json_file}...")
        
        try:
            with open(json_file, 'r') as f:
                self.annotations = json.load(f)
            print("Annotations loaded successfully.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {json_file}. Error: {str(e)}")

        if isinstance(self.annotations, dict):
            if 'annotations' in self.annotations:
                self.annotations = self.annotations['annotations']
            else:
                raise KeyError(f"Expected 'annotations' key not found in the JSON file: {json_file}")

        if not isinstance(self.annotations, list):
            raise TypeError(f"Annotations should be a list, but got {type(self.annotations)}")

        # Initialize image filenames and labels
        self.image_filenames = []
        self.image_labels = {}

        print("Processing annotations...")
        # Use tqdm to show progress for processing annotations
        for ann in tqdm(self.annotations, desc="Processing annotations", unit="annotation"):
            image_id = ann.get('image_id', None)
            category_id = ann.get('category_id', None)
            
            if image_id is None or category_id is None:
                raise KeyError(f"Annotation missing 'image_id' or 'category_id': {ann}")
            
            image_filename = f"image_{image_id}.jpg"
            self.image_filenames.append(image_filename)
            self.image_labels[image_filename] = category_id
        print(f"Processed {len(self.image_filenames)} annotations.")

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")  
            return None
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None
        
        label = self.image_labels.get(img_name, None)
        if label is None:
            print(f"Warning: Label not found for image {img_name}, skipping.")
            return None
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Creating dataset...")
dataset = ProcessedImagesDataset(images_dir=processed_images_dir, json_file=json_file, transform=transform)

# Efficient splitting using numpy's array slicing to avoid using index()
print("Splitting dataset into training and validation sets...")
image_filenames_array = np.array(dataset.image_filenames)

# Use train_test_split to get train and validation filenames
train_filenames, val_filenames = train_test_split(image_filenames_array, test_size=0.2, random_state=42)

# Show progress bar for dataset splitting
print(f"Splitting dataset into training ({len(train_filenames)}) and validation ({len(val_filenames)}) sets...")
train_indices = [np.where(image_filenames_array == f)[0][0] for f in tqdm(train_filenames, desc="Training Split", unit="file")]
val_indices = [np.where(image_filenames_array == f)[0][0] for f in tqdm(val_filenames, desc="Validation Split", unit="file")]

# Create subsets for training and validation
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Create DataLoader instances for both datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Data loaders created successfully.")

# Print sample batches for training
print("Starting to load and process training data...")
for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
    if images is None or labels is None:
        continue
    print(f"Batch {batch_idx+1} - Image Shape: {images.shape} - Labels: {labels}")
    break  # Only print the first batch

# Print sample batches for validation
print("Starting to load and process validation data...")
for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
    if images is None or labels is None:
        continue
    print(f"Batch {batch_idx+1} - Image Shape: {images.shape} - Labels: {labels}")
    break  # Only print the first batch

print("Program completed.")
