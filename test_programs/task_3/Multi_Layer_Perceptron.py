#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import json
import sys
import logging

#Libraries from PyTorch
from torchvision.io import read_image
import torch  # PyTorch's core library for building and training deep learning models
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.optim as optim  # Import optimization algorithms such as SGD, Adam
import torch.nn.functional as F  # Import functional utilities like activation functions
from torch.utils.data import Dataset, DataLoader, random_split # DataLoader for batching, and random_split for splitting data
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

#Libraries for data processing and visualization
#from matplotlib import pyplot as plt # For plotting graphs
import matplotlib.patches as patches # For bounding boxes
import numpy as np # For numerical operations
from PIL import Image
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For evaluation metrics
from tqdm import tqdm
sys.path.append(os.path.abspath('/home/rehan/Projects/Pytorch_Image_Classification/test_programs/task_2'))
from create_data_loaders import train_loader, val_loader  # Import DataLoader objects
print("data loader is")



# # In[2]:


# image_ids_annotations = defaultdict(list)

# # Load annotations
# path = './annotations/instances_train2017.json'
# file = open(path)
# anns = json.load(file)
# image_ids = list()

# # Add into datastructure
# for ann in anns['annotations']:
#     image_id = ann['image_id'] # Are integers
#     image_ids.append(image_id)
#     image_ids_annotations[image_id].append(ann)


# # In[3]:


# # Inicialization of the values per categorie  
# catergory_id_count = dict()
# for ann in anns['categories']:
#     catergory_id_count[ann['id']] = 0

# # Identification of the repetion per category 
# for ann in anns['annotations']:
#     category_id = ann['category_id']
#     catergory_id_count[category_id] += 1

# print(catergory_id_count)


# # In[4]:


# # Get mapping category_id to category name
# catergory_id_to_name = dict()
# for ann in anns['categories']:
#     catergory_id_to_name[ann['id']] = ann['name']

# print(catergory_id_to_name)


# # In[5]:


# # Creation of the Histogram 
# values = list(catergory_id_count.values())
# names = list(catergory_id_to_name.values())

# fig = plt.figure(figsize = (10, 5))

# # creating the bar plot
# plt.bar(names, values, color ='maroon', 
#         width = 0.4)
# plt.xticks(rotation=90)

# print(values)
# print(names)


# # In[6]:


# # Dataset loader for COCO
# class COCOMultiLabelDataset(Dataset):
#     def __init__(self, img_dir, ann_file, image_ids, transform=None): # Initialize the dataset with the images directory, the annotation file and transformations
#         self.coco = ann_file # Load COCO annotations from the provided annotations file
#         self.img_dir = img_dir  # Directory containing the images
#         self.transform = transform  # Transformations (e.g. resizing, cropping, ...) to apply to each image
#         self.ids = image_ids  # Get a list of image IDs from the dataset

#     # Return the number of images in the dataset
#     def __len__(self): 
#         return len(self.ids)

#     # Get an image and its corresponding labels, based on an index
#     def __getitem__(self, index):
#         img_id = self.ids[index] # Get the image ID corresponding to the given index
#         path = "0" * (12 - len(str(img_id))) +str(img_id) + ".jpg"
#         img_path = os.path.join(self.img_dir, path) # Create the full path to the image

#         # Load the image using PIL and convert it to RGB format
#         img = Image.open(img_path).convert("RGB")
#         if self.transform is not None: # If transformation is given, apply it to the image
#             img = self.transform(img)

#         # Get multi-label annotations
#         anns = self.coco[img_id]
#         labels = torch.zeros(90)  # Initialize a tensor of zeros for multi label classification (80 different classes in COCO)

#         # Iterate through each annotation to set the corresponding labels
#         for ann in anns:
#             category_id = ann['category_id'] # Extract the category ID from the annotation
#             labels[category_id-1] = 1.0

#         return img, labels # Return the transformed image and its multi-label tensor


# # In[7]:


# # Resizing of Images and Normalization
# img_dir = './train2017'

# img_transforms = transforms.Compose([
#         transforms.Resize((224, 224)), # Resize Images to a size of 24*24 Pixels
#         transforms.ToTensor(), 
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]) # Normalization using standard values for RGB images
#     ])

# train_data = COCOMultiLabelDataset(img_dir=img_dir,
#                                    ann_file=image_ids_annotations,
#                                    transform=img_transforms,
#                                    image_ids=image_ids) 


# # In[8]:


# train_size = int(0.8 * len(train_data))  # 90% of the data will be used for training
# val_size = int(0.1 * len(train_data))  # 10% of the data  will be used for validation
# test_size = len(train_data) - train_size - val_size # Remaining data will be used for test

# train_data, val_data, test_data = random_split(train_data, [train_size, val_size, test_size]) # Divide dataset into training, validation and test splits. 


# # In[9]:


# # DataLoader for the training set
# train_loader = DataLoader(
#     train_data,  # The training dataset
#     batch_size=16,  # Number of samples per batch during training
#     shuffle=True  # Shuffle the data at the start of every epoch for better generalization
# )

# # DataLoader for the validation set
# val_loader = DataLoader(
#     val_data,  # The validation dataset
#     batch_size=16,  # Same batch size as training
#     shuffle=False  # No need to shuffle validation data
# )

# # DataLoader for the validation set
# test_loader = DataLoader(
#     test_data,  # The validation dataset
#     batch_size=16,  # Same batch size as training
#     shuffle=False  # No need to shuffle validation data
# )

# # Printing of the sizes of the datasets
# print(f"Training set size: {train_size}")
# print(f"Validation set size: {val_size}")
# print(f"Test set size: {len(test_data)}")


# # In[10]:


# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

# Define the OOP Model class inheriting from nn.Module
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        logging.debug("Initializing CustomMLP model...")

        # First Layer
        self.fc1 = nn.Linear(224 * 224 * 3 , 512)  # Fully connected layer
        # Second Layer
        self.fc2 = nn.Linear(512, 512)  # Another fully connected layer
        # Output layer
        self.fc3 = nn.Linear(512, 90)  # Output layer
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        logging.debug("CustomMLP model initialized.")

    # Define the forward pass
    def forward(self, x):
        logging.debug(f"Input shape: {x.shape}")
        x = x.view(-1, 224 * 224 * 3)  # Flatten the input tensor
        x = self.relu(self.fc1(x))  # Pass through the first layer
        x = self.relu(self.fc2(x))  # Pass through the second layer
        x = self.sigmoid(self.fc3(x))  # Pass through the output layer
        logging.debug(f"Output shape: {x.shape}")
        return x

# Instantiate the model and move it to the appropriate device
oop_model = CustomMLP().to(device)
logging.debug("Model moved to device.")

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(oop_model.parameters(), lr=0.001)
logging.debug("Loss function and optimizer initialized.")

# Training function
def train_model(model, device, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    # Use tqdm for showing progress in training
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Zero gradients before backward pass
        output = model(data)  # Forward pass
        loss = loss_fn(output, target)  # Calculate loss

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate the loss

        if batch_idx % 50 == 0:  # Log every 50 batches
            logging.debug(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    logging.debug(f"Average training loss for this epoch: {avg_train_loss:.4f}")
    return avg_train_loss

# Validation function
def validate_model(model, device, val_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():  # No need for gradients in validation
        for data, target in tqdm(val_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)  # Forward pass
            loss = loss_fn(output, target)  # Calculate loss
            running_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_val_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    logging.debug(f"Validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_val_loss, accuracy

# Training and evaluation loop
def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        logging.debug(f"\nEpoch {epoch + 1}/{epochs}")
        logging.debug("-" * 50)

        # Training step
        train_loss = train_model(model, device, train_loader, optimizer, loss_fn)
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_accuracy = validate_model(model, device, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logging.debug(f"Training Loss: {train_loss:.4f}")
        logging.debug(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        torch.save(model.state_dict(), f"model_state_epoch_{epoch}.pt")  # Save model state

    return train_losses, val_losses, val_accuracies

# Setup logging to capture detailed debug information
logging.basicConfig(filename='train_validation_losses.log', level=logging.DEBUG)

# Example usage for 5 epochs
epochs = 5
train_losses, val_losses, val_accuracies = train_and_evaluate(
    oop_model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs=epochs
)

# After training, you can analyze the recorded losses and accuracies

# In[59]:


logging.debug('This is a debug message')


# In[54]:


logging.basicConfig(filename='example.log', level=logging.DEBUG)

