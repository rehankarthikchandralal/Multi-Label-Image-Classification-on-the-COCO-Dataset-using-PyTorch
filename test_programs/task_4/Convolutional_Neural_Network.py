#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import json
import logging

# Libraries from PyTorch
from torchvision.io import read_image
import torch  # PyTorch's core library for building and training deep learning models
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.optim as optim  # Import optimization algorithms such as SGD, Adam
import torch.nn.functional as F  # Import functional utilities like activation functions
from torch.utils.data import Dataset, DataLoader, random_split # DataLoader for batching, and random_split for splitting data
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

# Libraries for data processing and visualization
from matplotlib import pyplot as plt # For plotting graphs
import matplotlib.patches as patches # For bounding boxes
import numpy as np # For numerical operations
from PIL import Image
from collections import defaultdict
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For evaluation metrics

plt.rcParams['figure.figsize'] = (15, 15)

# In[2]:

image_ids_annotations = defaultdict(list)

# Load annotations
path = '/home/rehan/Projects/Pytorch_Image_Classification/coco/annotations/annotations/instances_train2017.json'
file = open(path)
anns = json.load(file)
image_ids = list()

# Add into datastructure
for ann in anns['annotations']:
    image_id = ann['image_id'] # Are integers
    image_ids.append(image_id)
    image_ids_annotations[image_id].append(ann)

# In[3]:

# Initialization of the values per category  
catergory_id_count = dict()
for ann in anns['categories']:
    catergory_id_count[ann['id']] = 0

# Identification of the repetition per category 
for ann in anns['annotations']:
    category_id = ann['category_id']
    catergory_id_count[category_id] += 1

print(catergory_id_count)

# In[4]:

# Get mapping category_id to category name
catergory_id_to_name = dict()
for ann in anns['categories']:
    catergory_id_to_name[ann['id']] = ann['name']

print(catergory_id_to_name)

# In[5]:

# Creation of the Histogram 
values = list(catergory_id_count.values())
names = list(catergory_id_to_name.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(names, values, color ='maroon', 
        width = 0.4)
plt.xticks(rotation=90)

print(values)
print(names)

# In[6]:

# Dataset loader for COCO
class COCOMultiLabelDataset(Dataset):
    def __init__(self, img_dir, ann_file, image_ids, transform=None): # Initialize the dataset with the images directory, the annotation file and transformations
        self.coco = ann_file # Load COCO annotations from the provided annotations file
        self.img_dir = img_dir  # Directory containing the images
        self.transform = transform  # Transformations (e.g. resizing, cropping, ...) to apply to each image
        self.ids = image_ids  # Get a list of image IDs from the dataset

    # Return the number of images in the dataset
    def __len__(self): 
        return len(self.ids)

    # Get an image and its corresponding labels, based on an index
# Get an image and its corresponding labels, based on an index
# Get an image and its corresponding labels, based on an index
    def __getitem__(self, index):
        img_id = self.ids[index]  # Get the image ID corresponding to the given index
        path = "0" * (12 - len(str(img_id))) + str(img_id) + ".jpg"
        img_path = os.path.join(self.img_dir, path)  # Create the full path to the image

        # Check if the image exists
        attempts = 0
        while not os.path.exists(img_path) and attempts < 10:  # Try up to 10 times to skip bad images
            print(f"Image {img_path} not found. Skipping.")
            index = (index + 1) % len(self.ids)  # Skip to next image
            img_id = self.ids[index]  # Get new image ID
            path = "0" * (12 - len(str(img_id))) + str(img_id) + ".jpg"
            img_path = os.path.join(self.img_dir, path)
            attempts += 1

        if attempts == 10:  # If unable to find a valid image after 10 tries
            print(f"Unable to find a valid image after 10 attempts. Returning empty tensor.")
            return torch.zeros(3, 224, 224), torch.zeros(90)  # Return a dummy image and label

        # Load the image using PIL and convert it to RGB format
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            return self.__getitem__((index + 1) % len(self.ids))  # This will just move to the next image

        if self.transform is not None:  # If transformation is given, apply it to the image
            img = self.transform(img)

        # Get multi-label annotations
        anns = self.coco[img_id]
        labels = torch.zeros(90)  # Initialize a tensor of zeros for multi-label classification (80 different classes in COCO)

        # Iterate through each annotation to set the corresponding labels
        for ann in anns:
            category_id = ann['category_id']  # Extract the category ID from the annotation
            labels[category_id-1] = 1.0

        return img, labels  # Return the transformed image and its multi-label tensor



# In[7]:

# Resizing of Images and Normalization
img_dir = '/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017'

img_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Resize Images to a size of 224*224 Pixels
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # Normalization using standard values for RGB images
    ])

train_data = COCOMultiLabelDataset(img_dir=img_dir,
                                   ann_file=image_ids_annotations,
                                   transform=img_transforms,
                                   image_ids=image_ids) 

# In[8]:

train_size = int(0.8 * len(train_data))  # 80% of the data will be used for training
val_size = int(0.1 * len(train_data))  # 10% of the data will be used for validation
test_size = len(train_data) - train_size - val_size # Remaining data will be used for test

train_data, val_data, test_data = random_split(train_data, [train_size, val_size, test_size]) # Divide dataset into training, validation and test splits. 

# In[9]:

# DataLoader for the training set
# Modify the subset size for quick testing
train_subset_size = int(0.8 * len(train_data))   # Use 1000 images for training, validation, and testing
val_subset_size = int(0.1 * len(train_data))
test_subset_size = len(train_data) - train_size - val_size
# Create a subset of the full dataset for training, validation, and testing
train_data = torch.utils.data.Subset(train_data, range(train_subset_size))
val_data = torch.utils.data.Subset(val_data, range(val_subset_size))
test_data = torch.utils.data.Subset(test_data, range(test_subset_size))

# DataLoader for the training set (with subset)
train_loader = DataLoader(
    train_data,  # The subset of the training dataset
    batch_size=16,  # Batch size for training
    shuffle=True  # Shuffle the data for better generalization
)

# DataLoader for the validation set (with subset)
val_loader = DataLoader(
    val_data,  # The subset of the validation dataset
    batch_size=16,  # Batch size for validation
    shuffle=False  # No need to shuffle validation data
)

# DataLoader for the test set (with subset)
test_loader = DataLoader(
    test_data,  # The subset of the test dataset
    batch_size=16,  # Batch size for testing
    shuffle=False  # No need to shuffle test data
)

# Printing the size of each dataset to ensure it's using a subset
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# In[10]:

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and output the result (True if a GPU is available, False otherwise)
torch.cuda.is_available()

# In[11]:

# Define the OOP Model class inheriting from nn.Module
# Define the CNN Model class inheriting from nn.Module
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()  # Call the parent class's constructor

        # Define the layers of the CNN
        # Convolutional Layer 1: 16 filters, kernel size 3x3, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # ReLU activation for Conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max Pooling Layer

        # Convolutional Layer 2: 32 filters, kernel size 3x3, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # ReLU activation for Conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max Pooling Layer

        # Convolutional Layer 3: 64 filters, kernel size 3x3, stride 1, padding 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()  # ReLU activation for Conv3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max Pooling Layer

        # Fully connected layer: input size depends on flattened dimensions, output size is 80
        self.fc = nn.Linear(64 * 28 * 28, 90)  # Calculate 28*28 from the input size (224x224) after 3 pooling layers

        # Output activation function
        self.sigmoid = nn.Sigmoid()

    # Define the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc(x)  # Fully connected layer

        return self.sigmoid(x)  # Sigmoid activation for multi-label classification

# Instantiate the CNN model
oop_model = CustomCNN().to(device)  # Move the model to the device (GPU/CPU)


# In[13]:

# Define Loss function
loss_fn = nn.BCELoss()  # Use normal Binary Cross-Entropy loss

# Define Optimizer
optimizer = optim.Adam(oop_model.parameters(), lr=0.001)

# In[14]:

# Modified training loop to return average training loss for each epoch and resume from a specific checkpoint
# Modified training loop to include validation loss calculation
def train_model(model, device, train_loader, val_loader, optimizer, loss_fn, start_epoch=0, num_epochs=15, checkpoint_path="model_checkpoint.pth"):
    model.train()  # Set the model to training mode

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Open log file in append mode
    with open('training_validation_log_cnn.txt', 'a') as log_file:
        # Loop through epochs starting from the specified epoch
        for epoch in range(start_epoch, num_epochs):
            running_loss = 0.0

            # Training step
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch")):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", unit="batch"):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Log the training and validation losses
            log_file.write(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            log_file.flush()

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_name = f"model_checkpoint_epoch_{epoch+1}.pth"
                print(f"Saving checkpoint: {checkpoint_name}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_name)

    print("Training complete.")


# In[15]:

# Call the function to train the model
train_model(oop_model, device, train_loader, val_loader, optimizer, loss_fn, num_epochs=15)




