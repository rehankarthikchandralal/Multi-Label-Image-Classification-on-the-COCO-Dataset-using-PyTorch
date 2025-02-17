#!/usr/bin/env python
# coding: utf-8

import os          # Provides functions to interact with the operating system
import json        # For handling JSON data
import logging     # For logging messages

# Libraries from PyTorch
from torchvision.io import read_image       # For reading image data
import torch                                # Core PyTorch library
import torch.nn as nn                      # Neural network module (layers, activation, etc.)
import torch.optim as optim                # Optimizers such as SGD, Adam
import torch.nn.functional as F            # Functional utilities (activation funcs, etc.)
from torch.utils.data import Dataset, DataLoader, random_split  # Data management classes
from torchvision import datasets, models, transforms            # Popular datasets, model architectures, transformations
from torchvision.models import ResNet50_Weights, resnet50       # ResNet50 model and its weights
from pycocotools.coco import COCO                              # COCO dataset utilities
from torch.nn.utils.rnn import pad_sequence                    # Utility for padding sequences

# Libraries for data processing and visualization
from matplotlib import pyplot as plt       # Plotting library
import matplotlib.patches as patches       # For drawing bounding boxes
import numpy as np                         # Numerical operations
from PIL import Image                      # Image handling
from collections import defaultdict        # For convenient dict subclasses

from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)  # Evaluation metrics

plt.rcParams['figure.figsize'] = (15, 15)  # Set default figure size

import pandas as pd       # Data analysis library
import seaborn as sn       # Data visualization built on matplotlib

# ------------------------------------------------------------------------------
# Step 1: Load annotations and build data structures
# ------------------------------------------------------------------------------
image_ids_annotations = defaultdict(list)     # Map from image ID -> list of annotations

path = './annotations/instances_train2017_mod.json'  # Path to the modified COCO annotations
file = open(path)                              # Open the JSON file
anns = json.load(file)                         # Load JSON into a dict
image_ids = list()                             # Will store image IDs

for ann in anns['annotations']:               # Iterate over each annotation entry
    image_id = ann['image_id']                # Unique integer ID for each image
    image_ids.append(image_id)                # Collect image ID
    image_ids_annotations[image_id].append(ann)  # Append the annotation to that image's list

# ------------------------------------------------------------------------------
# Step 2: Count the occurrences of each category
# ------------------------------------------------------------------------------
catergory_id_count = dict()                   # Will hold category_id -> count
contador = list(range(1,81))                  # Category IDs from 1 to 80

for a in contador:
    catergory_id_count[a] = 0                 # Initialize count to zero

for ann in anns['annotations']:              # Go through all annotations
    category_id = ann['category_id']          # Extract category_id
    catergory_id_count[category_id] += 1      # Increment its counter

print(catergory_id_count)

# ------------------------------------------------------------------------------
# Step 3: Map category IDs to category names
# ------------------------------------------------------------------------------
catergory_id_to_name = dict()                 # Will map category ID -> category name
for index, cat in enumerate(anns['categories']):
    catergory_id_to_name[cat['id']] = cat['name']

print(anns['categories'])
print(catergory_id_count)
print(catergory_id_to_name)

# ------------------------------------------------------------------------------
# Step 4: Plot a histogram of category frequencies
# ------------------------------------------------------------------------------
values = list(catergory_id_count.values())    # List of counts
classes = list(catergory_id_to_name.values()) # List of category names
print(len(values))
print(len(classes))

fig = plt.figure(figsize=(10, 5))             # Create a figure
plt.bar(classes, values, color='maroon', width=0.4)  # Plot bar chart
plt.xticks(rotation=90)                       # Rotate x-axis labels for readability
print(values)

# ------------------------------------------------------------------------------
# Step 5: Custom Dataset class to handle COCO detection format
# ------------------------------------------------------------------------------
class CocoDetection(Dataset):
    """
    MS Coco Detection Dataset.
    Args:
        root (str): Root directory of images.
        annFile (str): Path to the JSON annotation file.
        transform (callable): Optional transform to be applied on a sample.
        target_transform (callable): Optional transform to be applied on the target.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root                           # Directory with images
        self.coco = COCO(annFile)                  # Initialize COCO API
        self.ids = list(self.coco.imgs.keys())     # Get all image IDs
        self.transform = transform                 # Optional transforms on images
        self.target_transform = target_transform   # Optional transforms on targets

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]                   # Select image ID
        ann_ids = coco.getAnnIds(imgIds=img_id)    # Get annotation IDs for this image
        target = coco.loadAnns(ann_ids)            # Load annotation details

        path = coco.loadImgs(img_id)[0]['file_name']  # Retrieve image file name
        img = Image.open(os.path.join(self.root, path)).convert('RGB')  # Open and convert to RGB

        if self.transform is not None:             # Apply image transforms if specified
            img = self.transform(img)

        labels = torch.zeros(80)                   # Initialize vector for 80 categories

        for ann in target:                         # For each annotation
            category_id = ann['category_id']       # Category ID
            labels[category_id - 1] = 1.0          # Mark as 1 for that category

        return img, labels                         # Return image tensor + multi-hot labels

    def __len__(self):
        return len(self.ids)                       # Total number of images

    def __repr__(self):
        # String representation of this dataset
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += f'    Number of datapoints: {self.__len__()}\n'
        fmt_str += f'    Root Location: {self.root}\n'
        tmp = '    Transforms (if any): '
        fmt_str += f'{tmp}{self.transform.__repr__().replace("\n", "\n" + " "*len(tmp))}\n'
        tmp = '    Target Transforms (if any): '
        fmt_str += f'{tmp}{self.target_transform.__repr__().replace("\n", "\n" + " "*len(tmp))}'
        return fmt_str

# ------------------------------------------------------------------------------
# Step 6: Define transforms and instantiate dataset
# ------------------------------------------------------------------------------
img_dir = './train2017'    # Directory containing training images

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to 224x224
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

train_data = CocoDetection(
    root=img_dir,
    annFile=path,
    transform=img_transforms
)

# ------------------------------------------------------------------------------
# Step 7: Split the dataset into train/val/test
# ------------------------------------------------------------------------------
train_size = int(0.8 * len(train_data))  # 80% for training
val_size = int(0.1 * len(train_data))    # 10% for validation
test_size = len(train_data) - train_size - val_size  # Remaining 10% for test

train_data, val_data, test_data = random_split(
    train_data, [train_size, val_size, test_size]
)

# ------------------------------------------------------------------------------
# Step 8: Quick check of a test image and its labels
# ------------------------------------------------------------------------------
img_test, target_test = test_data.__getitem__(2)   # Get the 3rd sample in test set
print(target_test)                                 # Print label vector
print(img_test.size())                             # Print the shape of the image tensor

image_numpy = img_test.permute(1, 2, 0).numpy()    # Rearrange dimensions for plotting
image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())  # Normalize for display

plt.imshow(image_numpy)                            # Display the image
plt.axis('off')                                    # Hide axes
plt.show()

target_test = target_test.numpy()                  # Convert label tensor to NumPy array
for i in range(len(classes)):                      # Print out categories present in the sample
    if target_test[i] == 1:
        print(classes[i])

# ------------------------------------------------------------------------------
# Step 9: Create DataLoaders for training, validation, and test
# ------------------------------------------------------------------------------
train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
)

print(f"Training set size: {train_size}")
print(f"Validation set size: {val_size}")
print(f"Test set size: {len(test_data)}")

# ------------------------------------------------------------------------------
# Step 10: Check GPU availability
# ------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()  # Returns True if GPU is available, else False

# ------------------------------------------------------------------------------
# Step 11: Define a simple MLP Model
# ------------------------------------------------------------------------------
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()         # Init parent class
        self.fc1 = nn.Linear(224 * 224 * 3, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 512)            # Second fully connected layer
        self.fc3 = nn.Linear(512, 90)             # Output layer (80 categories + extra if needed)
        self.relu = nn.ReLU()                     # ReLU activation
        self.sigmoid = nn.Sigmoid()               # Sigmoid for multi-label classification

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)             # Flatten the image batch
        x = F.relu(self.fc1(x))                   # FC1 + ReLU
        x = F.relu(self.fc2(x))                   # FC2 + ReLU
        return self.sigmoid(self.fc3(x))          # Output layer + Sigmoid

oop_model = CustomMLP().to(device)                # Instantiate and move to GPU if available
print(oop_model)

# ------------------------------------------------------------------------------
# Step 12: Define loss function and optimizer
# ------------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()                   # Cross-entropy for multi-class classification
optimizer = optim.Adam(oop_model.parameters(), lr=0.001)  # Adam optimizer

# ------------------------------------------------------------------------------
# Step 13: Training function
# ------------------------------------------------------------------------------
def train_model(model, device, train_loader, optimizer, loss_fn):
    model.train()                                  # Set to training mode
    running_loss = 0.0                             # Track total loss

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to device
        optimizer.zero_grad()                               # Zero out gradients
        output = model(data)                                # Forward pass
        loss = loss_fn(output, target)                      # Compute loss
        loss.backward()                                     # Backprop
        optimizer.step()                                    # Update weights
        running_loss += loss.item()                         # Accumulate loss

    avg_train_loss = running_loss / len(train_loader)        # Average loss
    return avg_train_loss

# Example usage (uncomment to run one training iteration)
# train_loss = train_model(oop_model, device, train_loader, optimizer, loss_fn)

# ------------------------------------------------------------------------------
# Step 14: Validation function
# ------------------------------------------------------------------------------
def validate_model(model, device, val_loader, loss_fn):
    model.eval()                                   # Set to eval mode
    running_loss = 0.0

    with torch.no_grad():                          # No gradient computation
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss = loss_fn(output, target)
            running_loss += val_loss.item()

    avg_val_loss = running_loss / len(val_loader)
    return avg_val_loss

# ------------------------------------------------------------------------------
# Step 15: Test function
# ------------------------------------------------------------------------------
def test_model(model, device, test_loader, loss_fn):
    model.eval()                                   
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = loss_fn(output, target)
            running_loss += test_loss.item()

    avg_test_loss = running_loss / len(test_loader)
    return avg_test_loss

# ------------------------------------------------------------------------------
# Step 16: Combined train & evaluation loop
# ------------------------------------------------------------------------------
def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, epochs=10):
    train_losses = []  # Track training losses
    val_losses = []    # Track validation losses

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        logging.debug(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_model(model, device, train_loader, optimizer, loss_fn)
        train_losses.append(train_loss)

        val_loss = validate_model(model, device, val_loader, loss_fn)
        val_losses.append(val_loss)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        logging.debug(f"Training Loss: {train_loss:.4f}")
        logging.debug(f"Validation Loss: {val_loss:.4f}")

        # Save model state after each epoch
        torch.save(model.state_dict(), "model_state_epoch_" + str(epoch+10) + ".pt")

    return train_losses, val_losses

# Example usage (commented out):
# epochs = 10  
# logging.basicConfig(filename='train_validation_losses.log', level=logging.DEBUG)

# train_losses, val_losses, val_accuracies, test_losses, test_accuracies = train_and_evaluate(
#     oop_model,
#     device,
#     train_loader,
#     val_loader,
#     test_loader,
#     optimizer,
#     loss_fn,
#     epochs=epochs
# )

# ------------------------------------------------------------------------------
# Step 17: Load a saved model state (example)
# ------------------------------------------------------------------------------
model = CustomMLP().to(device)
model.load_state_dict(torch.load(
    'model_checkpoint.pth', 
    weights_only=True, 
    map_location=torch.device('cpu')
)['model_state_dict'])
model.eval()  # Set to evaluation mode

# ------------------------------------------------------------------------------
# TASK 5: Evaluation - counting TP, FP, TN, FN for each class
# ------------------------------------------------------------------------------
true_positive = {classname: 0 for classname in classes}
false_positive = {classname: 0 for classname in classes}
false_negative = {classname: 0 for classname in classes}
true_negative = {classname: 0 for classname in classes}

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = target[0]
        output = torch.round(output)[0]
        for index, (label, prediction) in enumerate(zip(target, output)):
            if label == prediction:
                if prediction == 1.:
                    true_positive[classes[index]] += 1
                else:
                    true_negative[classes[index]] += 1
            else:
                if prediction == 1:
                    false_positive[classes[index]] += 1
                else:
                    false_negative[classes[index]] += 1

cat = 'person'  # Example category
tp = true_positive[cat] / (true_positive[cat] + false_negative[cat]) if (true_positive[cat] + false_negative[cat]) > 0 else 0
fn = false_negative[cat] / (true_positive[cat] + false_negative[cat]) if (true_positive[cat] + false_negative[cat]) > 0 else 0
tn = true_negative[cat] / (true_negative[cat] + false_positive[cat]) if (true_negative[cat] + false_positive[cat]) > 0 else 0
fp = false_positive[cat] / (false_positive[cat] + true_negative[cat]) if (false_positive[cat] + true_negative[cat]) > 0 else 0

print(tp, fn, fp, tn)
print(true_positive[cat], false_negative[cat], false_positive[cat], true_negative[cat])

accuracy_MLP = (true_positive[cat] + true_negative[cat]) / (
    true_positive[cat] + 
    true_negative[cat] + 
    false_positive[cat] + 
    false_negative[cat]
) if (true_positive[cat] + true_negative[cat] + false_positive[cat] + false_negative[cat]) > 0 else 0

precision_MLP = true_positive[cat] / (true_positive[cat] + false_positive[cat]) if (true_positive[cat] + false_positive[cat]) > 0 else 0
recall_MLP = true_positive[cat] / (true_positive[cat] + false_negative[cat]) if (true_positive[cat] + false_negative[cat]) > 0 else 0
f1_MLP = 2 * (precision_MLP * recall_MLP) / (precision_MLP + recall_MLP) if (precision_MLP + recall_MLP) > 0 else 0

print(accuracy_MLP, precision_MLP, recall_MLP, f1_MLP)

# ------------------------------------------------------------------------------
# TASK 6: Fine-tuning a pretrained ResNet50
# ------------------------------------------------------------------------------
pretrained_model = models.resnet50(weights='IMAGENET1K_V1').to(device)
for param in pretrained_model.parameters():
    param.requires_grad = False

pretrained_model.fc = nn.Sequential(
    nn.Linear(2048, 80).to(device),
    nn.Sigmoid()
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

# Example: train_and_evaluate call
# train_losses, val_losses = train_and_evaluate(
#     pretrained_model, 
#     device, 
#     train_loader, 
#     val_loader, 
#     optimizer, 
#     loss_fn, 
#     epochs=5
# )

pretrained_model.load_state_dict(torch.load('model_state_epoch_14.pt', 
                                            weights_only=True, 
                                            map_location=torch.device('cpu')))
pretrained_model.eval()

true_positive_pt_model = {classname: 0 for classname in classes}
false_positive_pt_model = {classname: 0 for classname in classes}
false_negative_pt_model = {classname: 0 for classname in classes}
true_negative_pt_model = {classname: 0 for classname in classes}

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = pretrained_model(data)
        target = target[0]
        output = torch.round(output)[0]
        for index, (label, prediction) in enumerate(zip(target, output)):
            if label == prediction:
                if prediction == 1.:
                    true_positive_pt_model[classes[index]] += 1
                else:
                    true_negative_pt_model[classes[index]] += 1
            else:
                if prediction == 1:
                    false_positive_pt_model[classes[index]] += 1
                else:
                    false_negative_pt_model[classes[index]] += 1

cat = 'chair'  # Example category
tp_rn = true_positive_pt_model[cat] / (true_positive_pt_model[cat] + false_negative_pt_model[cat]) if (true_positive_pt_model[cat] + false_negative_pt_model[cat]) > 0 else 0
fn_rn = false_negative_pt_model[cat] / (true_positive_pt_model[cat] + false_negative_pt_model[cat]) if (true_positive_pt_model[cat] + false_negative_pt_model[cat]) > 0 else 0
tn_rn = true_negative_pt_model[cat] / (true_negative_pt_model[cat] + false_positive_pt_model[cat]) if (true_negative_pt_model[cat] + false_positive_pt_model[cat]) > 0 else 0
fp_rn = false_positive_pt_model[cat] / (false_positive_pt_model[cat] + true_negative_pt_model[cat]) if (false_positive_pt_model[cat] + true_negative_pt_model[cat]) > 0 else 0

print(tp_rn, fn_rn, fp_rn, tn_rn)
print(true_positive_pt_model[cat], false_negative_pt_model[cat], false_positive_pt_model[cat], true_negative_pt_model[cat])

accuracy_RN = (true_positive_pt_model[cat] + true_negative_pt_model[cat]) / (
    true_positive_pt_model[cat] + 
    true_negative_pt_model[cat] + 
    false_positive_pt_model[cat] + 
    false_negative_pt_model[cat]
) if (true_positive_pt_model[cat] + true_negative_pt_model[cat] + false_positive_pt_model[cat] + false_negative_pt_model[cat]) > 0 else 0

precision_RN = true_positive_pt_model[cat] / (true_positive_pt_model[cat] + false_positive_pt_model[cat]) if (true_positive_pt_model[cat] + false_positive_pt_model[cat]) > 0 else 0
recall_RN = true_positive_pt_model[cat] / (true_positive_pt_model[cat] + false_negative_pt_model[cat]) if (true_positive_pt_model[cat] + false_negative_pt_model[cat]) > 0 else 0
f1_RN = 2*(precision_RN*recall_RN)/(precision_RN+recall_RN) if (precision_RN+recall_RN) > 0 else 0

print(accuracy_RN, precision_RN, recall_RN, f1_RN)

# ------------------------------------------------------------------------------
# TASK 7: Simple demonstration of plotting losses
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

epochs = 16
training_loss = np.array([10.8224, 10.7825, 10.7734, 10.7660, 
                          10.7600, 10.7563, 10.7531, 10.7496, 
                          10.7470, 10.7381, 10.7349, 10.7290, 
                          10.7293, 10.7340, 10.7294, 10.7304])
validation_loss = np.array([10.7218, 10.7155, 10.7144, 10.6994, 
                            10.7135, 10.7013, 10.7012, 10.7042, 
                            10.6865, 10.6926, 10.6818, 10.6877, 
                            10.6823, 10.6851, 10.6869, 10.6916])

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), training_loss, marker='o', label='Training Loss')
plt.plot(range(1, epochs + 1), validation_loss, marker='s', label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------------------------------------------
# Additional CNN Model example (CustomCNN) and evaluation code
# ------------------------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 pooling layers on a 224x224 input => (224 / 2 / 2 / 2) = 28
        self.fc = nn.Linear(64 * 28 * 28, 90)
        self.sigmoid = nn.Sigmoid()

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

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return self.sigmoid(x)

cnn_model = CustomCNN().to(device)
cnn_model.load_state_dict(torch.load('model_checkpoint_epoch_15_cnn.pth', 
                                     weights_only=True, 
                                     map_location=torch.device('cpu'))['model_state_dict'])
cnn_model.eval()

# Prepare to count predictions
true_positive_cnn_model = {classname: 0 for classname in classes}
false_positive_cnn_model = {classname: 0 for classname in classes}
false_negative_cnn_model = {classname: 0 for classname in classes}
true_negative_cnn_model = {classname: 0 for classname in classes}

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = cnn_model(data)
        target = target[0]
        output = torch.round(output)[0]

        for index, (label, prediction) in enumerate(zip(target, output)):
            if label == prediction:
                if prediction == 1.:
                    true_positive_cnn_model[classes[index]] += 1
                else:
                    true_negative_cnn_model[classes[index]] += 1
            else:
                if prediction == 1:
                    false_positive_cnn_model[classes[index]] += 1
                else:
                    false_negative_cnn_model[classes[index]] += 1

cat = 'chair'
tp_rn = true_positive_cnn_model[cat] / (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) if (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) > 0 else 0
fn_rn = false_negative_cnn_model[cat] / (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) if (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) > 0 else 0
tn_rn = true_negative_cnn_model[cat] / (true_negative_cnn_model[cat] + false_positive_cnn_model[cat]) if (true_negative_cnn_model[cat] + false_positive_cnn_model[cat]) > 0 else 0
fp_rn = false_positive_cnn_model[cat] / (false_positive_cnn_model[cat] + true_negative_cnn_model[cat]) if (false_positive_cnn_model[cat] + true_negative_cnn_model[cat]) > 0 else 0

print(tp_rn, fn_rn, fp_rn, tn_rn)
print(true_positive_cnn_model[cat], false_negative_cnn_model[cat], 
      false_positive_cnn_model[cat], true_negative_cnn_model[cat])

accuracy_RN = (true_positive_cnn_model[cat] + true_negative_cnn_model[cat]) / (
    true_positive_cnn_model[cat] + 
    true_negative_cnn_model[cat] + 
    false_positive_cnn_model[cat] + 
    false_negative_cnn_model[cat]
) if (true_positive_cnn_model[cat] + true_negative_cnn_model[cat] + false_positive_cnn_model[cat] + false_negative_cnn_model[cat]) > 0 else 0

precision_RN = true_positive_cnn_model[cat] / (true_positive_cnn_model[cat] + false_positive_cnn_model[cat]) if (true_positive_cnn_model[cat] + false_positive_cnn_model[cat]) > 0 else 0
recall_RN = true_positive_cnn_model[cat] / (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) if (true_positive_cnn_model[cat] + false_negative_cnn_model[cat]) > 0 else 0
f1_RN = 2 * (precision_RN * recall_RN) / (precision_RN + recall_RN) if (precision_RN + recall_RN) > 0 else 0

print(accuracy_RN, precision_RN, recall_RN, f1_RN)

