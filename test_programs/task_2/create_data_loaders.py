import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import pickle
from tqdm import tqdm

# Define paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"
train_json_file = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/instances_train.json"
val_json_file = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/instances_val.json"
data_loader_save_dir = "/home/rehan/Projects/Pytorch_Image_Classification/dataloaders"

# Ensure directories and files exist
if not os.path.exists(processed_images_dir):
    raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

if not os.path.exists(train_json_file):
    raise FileNotFoundError(f"Train JSON annotations file not found: {train_json_file}")

if not os.path.exists(val_json_file):
    raise FileNotFoundError(f"Validation JSON annotations file not found: {val_json_file}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ProcessedImagesDataset(Dataset):
    def __init__(self, images_dir, json_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load annotations from JSON file
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        if isinstance(self.annotations, dict) and 'annotations' in self.annotations:
            self.annotations = self.annotations['annotations']

        self.image_filenames = []
        self.image_labels = {}

        # Process annotations
        for ann in tqdm(self.annotations, desc="Processing annotations", unit="annotation"):
            image_id = ann.get('image_id')
            category_id = ann.get('category_id')

            if image_id is None or category_id is None:
                continue  # Skip invalid annotations

            image_filename = f"{str(image_id).zfill(12)}.jpg"  # 12 digits padded for the processed images
            self.image_filenames.append(image_filename)
            self.image_labels[image_filename] = category_id

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
            return None  # Skip invalid sample

        image = Image.open(img_path).convert('RGB')
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


# Create DataLoader instances for train and validation
def create_dataloaders():
    # Create Dataset for train and validation
    train_dataset = ProcessedImagesDataset(images_dir=processed_images_dir, json_file=train_json_file, transform=transform)
    val_dataset = ProcessedImagesDataset(images_dir=processed_images_dir, json_file=val_json_file, transform=transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # Save the DataLoader objects
    with open(os.path.join(data_loader_save_dir, "train_loader.pkl"), "wb") as f:
        pickle.dump(train_loader, f)

    with open(os.path.join(data_loader_save_dir, "val_loader.pkl"), "wb") as f:
        pickle.dump(val_loader, f)

    print("Data loaders created and saved successfully.")

    return train_loader, val_loader


if __name__ == "__main__":
    # Create the DataLoader objects
    train_loader, val_loader = create_dataloaders()

    # Optionally, process a batch of data to verify the loaders
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
