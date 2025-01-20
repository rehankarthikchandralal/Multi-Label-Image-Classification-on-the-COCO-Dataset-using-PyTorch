import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import torch
from PIL import Image
import json
# Define paths
processed_images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"
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

file_path = "/home/rehan/Projects/Pytorch_Image_Classification/coco/annotations/annotations/instances_train2017.json"

with open(file_path, "r") as f:
    data = json.load(f)

print(data.keys())
if "categories" not in data:
    print("Categories key is missing!")
else:
    print("Categories:", data["categories"])

# Define transformations
# Define transformations (remove unnecessary ones)
transform = transforms.Compose([
    # Random crop and resize to 224x224
    transforms.RandomResizedCrop(224),  # Randomly crop and resize images to 224x224
    
    # Random horizontal flip for data augmentation
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    
    # Random rotation for data augmentation
    transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
    
    # Convert image to Tensor
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    
    # Normalize the image with mean and std (this will depend on your dataset)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # These are ImageNet statistics
])



# Dataset loader for COCO
class COCOMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())
        self.cat_to_contiguous = {cat['id']: idx for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Apply transformations if necessary
        if self.transform:
            img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        labels = torch.zeros(80)
        for ann in anns:
            category_id = ann['category_id']
            if category_id in self.cat_to_contiguous:
                contiguous_id = self.cat_to_contiguous[category_id]
                labels[contiguous_id] = 1.0
        return img, labels



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
    train_dataset = COCOMultiLabelDataset(img_dir=processed_images_dir, ann_file=train_json_file, transform=transform)
    val_dataset = COCOMultiLabelDataset(img_dir=processed_images_dir, ann_file=val_json_file, transform=transform)

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
            print(f"Batch {batch_idx+1}/{total_batches} - Image Shape: {images.shape} - Labels Shape: {labels.shape}")
            break  # Only process the first batch to keep it fast

    process_data(train_loader, "Training")
    process_data(val_loader, "Validation")

    print("Program completed.")
