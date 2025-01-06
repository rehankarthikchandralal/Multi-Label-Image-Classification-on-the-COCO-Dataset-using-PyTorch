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
split_save_dir = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/splits"

# Ensure directories and files exist
if not os.path.exists(processed_images_dir):
    raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON annotations file not found: {json_file}")

os.makedirs(split_save_dir, exist_ok=True)

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

        if isinstance(self.annotations, dict) and 'annotations' in self.annotations:
            self.annotations = self.annotations['annotations']

        if not isinstance(self.annotations, list):
            raise TypeError(f"Annotations should be a list, but got {type(self.annotations)}")

        self.image_filenames = []
        self.image_labels = {}

        print("Processing annotations...")
        for ann in tqdm(self.annotations, desc="Processing annotations", unit="annotation"):
            image_id = ann.get('image_id')
            category_id = ann.get('category_id')

            if image_id is None or category_id is None:
                continue  # Skip invalid annotations

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
            return None  # Skip invalid sample

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None  # Skip invalid sample

        label = self.image_labels.get(img_name)
        if label is None:
            print(f"Warning: Label not found for image {img_name}, skipping.")
            return None  # Skip invalid sample

        if self.transform:
            image = self.transform(image)

        return image, label

# Custom collate function to handle None values
def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data._utils.collate.default_collate(batch)

# Function to load or create splits
def get_splits(dataset, split_save_dir, test_size=0.2, random_state=42):
    train_split_file = os.path.join(split_save_dir, "train_split.npy")
    val_split_file = os.path.join(split_save_dir, "val_split.npy")

    if os.path.exists(train_split_file) and os.path.exists(val_split_file):
        print("Loading saved splits...")
        train_filenames = np.load(train_split_file, allow_pickle=True).tolist()
        val_filenames = np.load(val_split_file, allow_pickle=True).tolist()
        print("Splits loaded successfully.")
    else:
        print("Splits not found. Splitting dataset into training and validation sets...")
        train_filenames, val_filenames = train_test_split(
            dataset.image_filenames, test_size=test_size, random_state=random_state
        )
        np.save(train_split_file, train_filenames)
        np.save(val_split_file, val_filenames)
        print("Splits saved successfully.")

    return train_filenames, val_filenames

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Creating dataset...")
dataset = ProcessedImagesDataset(images_dir=processed_images_dir, json_file=json_file, transform=transform)

# Load or create splits
train_filenames, val_filenames = get_splits(dataset, split_save_dir)

# Convert filenames to indices
image_filenames_array = np.array(dataset.image_filenames)
train_indices = [np.where(image_filenames_array == f)[0][0] for f in tqdm(train_filenames, desc="Finding train indices", unit="file")]
val_indices = [np.where(image_filenames_array == f)[0][0] for f in tqdm(val_filenames, desc="Finding validation indices", unit="file")]

# Create DataLoader instances
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

print("Data loaders created successfully.")

# Load and process data
def process_data(loader, desc):
    total_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=desc, total=total_batches, ncols=100)):
        if images is None or labels is None:
            continue
        print(f"Batch {batch_idx+1}/{total_batches} - Image Shape: {images.shape} - Labels: {labels}")
        break  # Only process the first batch to keep it fast

process_data(train_loader, "Training")
process_data(val_loader, "Validation")

print("Program completed.")
