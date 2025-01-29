#!/usr/bin/env python
# coding: utf-8

# In[2]:

# Core libraries
import os
import json
import logging
from collections import defaultdict
from tqdm import tqdm

# PyTorch and torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.patches as patches

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, multilabel_confusion_matrix
)

# Keep original imports
plt.rcParams['figure.figsize'] = (15, 15)

# Load test annotations
test_annotations_path = '/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/instances_test.json'
file = open(test_annotations_path)
anns_test = json.load(file)
image_ids_test = list()
image_ids_annotations_test = defaultdict(list)

# Add into data structure for test
for ann in anns_test['annotations']:
    image_id = ann['image_id']
    image_ids_test.append(image_id)
    image_ids_annotations_test[image_id].append(ann)

# In[6]:
# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loader for COCO Test Set
class COCOMultiLabelDatasetTest(Dataset):
    def __init__(self, img_dir, ann_file, image_ids, transform=None):
        self.coco = ann_file
        self.img_dir = img_dir
        self.transform = transform
        self.ids = image_ids

    def __len__(self):
        return len(self.ids)

# Get an image and its corresponding labels, based on an index
    def __getitem__(self, index):
        img_id = self.ids[index]  # Get the image ID corresponding to the given index
        path = "0" * (12 - len(str(img_id))) + str(img_id) + ".jpg"
        img_path = os.path.join(self.img_dir, path)  # Create the full path to the image

        # Check if the image exists and skip if not
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
            return torch.zeros(3, 224, 224), torch.zeros(90)  # Return a dummy image and label

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

# Image transformations for test data
img_dir_test = '/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017'
img_transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

test_data = COCOMultiLabelDatasetTest(img_dir=img_dir_test,
                                      ann_file=image_ids_annotations_test,
                                      transform=img_transforms_test,
                                      image_ids=image_ids_test)

# DataLoader for the test set
test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
)

print(f"Test set size: {len(test_data)}")

# In[10]:

# Define the CNN Model class
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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.sigmoid(x)

# Instantiate and load the trained model
trained_model_path = '/home/rehan/Projects/Pytorch_Image_Classification/test_programs/task_5/model_checkpoint_epoch_15.pth'
model = CustomCNN().to(device)
checkpoint = torch.load(trained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# In[12]:

# Evaluate model on the test split
def evaluate_model_with_per_label_metrics(model, test_loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating Model", unit="batch"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            predictions = (output > 0.5).float()  # Threshold for multi-label classification

            y_true.append(target.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    overall_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    # Calculate per-label metrics
    label_accuracies = (y_true == y_pred).mean(axis=0)
    label_precisions = precision_score(y_true, y_pred, average=None, zero_division=1)
    label_recalls = recall_score(y_true, y_pred, average=None, zero_division=1)
    label_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=1)

    # Print per-label metrics
    print("\nPer-Label Metrics:")
    for i, (acc, prec, rec, f1) in enumerate(zip(label_accuracies, label_precisions, label_recalls, label_f1_scores)):
        print(f"Label {i + 1}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")

    return y_true, y_pred, label_accuracies, label_precisions, label_recalls, label_f1_scores

# In[14]:

# Call the updated evaluation function


# In[13]:

# Generate confusion matrix


def plot_multilabel_confusion_matrices(y_true, y_pred, class_names, normalize=False, cmap=plt.cm.Blues):
    """
    Plot confusion matrices for specific class names in a multilabel classification task.
    Each section of the confusion matrix is annotated as TP, FP, TN, or FN.
    This function plots confusion matrices for label 1 (person), label 15 (bench), and label 87 (scissors).
    """

    # Compute confusion matrices for each class
    cm_list = multilabel_confusion_matrix(y_true, y_pred)

    # Select class names and confusion matrices for label 1 (person), label 15 (bench), and label 87 (scissors)
    selected_classes = [class_names[1], class_names[15], class_names[80]]
    selected_indices = [0, 14, 70]  # Indices for label 1, label 15, and label 87 (0-based indexing)
    selected_cm_list = [cm_list[i] for i in selected_indices]

    # Plot confusion matrices for the selected class names
    num_classes = len(selected_classes)
    plt.figure(figsize=(10, num_classes * 4))  # Adjust figsize for readability

    for i, (class_name, cm) in enumerate(zip(selected_classes, selected_cm_list)):
        # Normalize if needed and avoid division by zero
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm = np.divide(cm.astype('float'), row_sums, where=row_sums != 0)
                cm = np.nan_to_num(cm)  # Replace NaNs with zeros for classes with no samples

        # Annotate the sections of the confusion matrix
        labels = np.array([
            ["TN", "FP"],
            ["FN", "TP"]
        ])
        section_values = np.array([
            [cm[0, 0], cm[0, 1]],
            [cm[1, 0], cm[1, 1]]
        ])

        # Combine labels with values for annotation
        annot = np.empty_like(section_values, dtype=object)
        for j in range(section_values.shape[0]):
            for k in range(section_values.shape[1]):
                annot[j, k] = f"{labels[j, k]}\n{section_values[j, k]:.2f}" if normalize else f"{labels[j, k]}\n{int(section_values[j, k])}"

        # Heatmap visualization with larger font size
        plt.subplot(1, num_classes, i + 1)  # Display matrices in a single row for the selected classes
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                    xticklabels=['Not ' + class_name, class_name],
                    yticklabels=['Not ' + class_name, class_name],
                    annot_kws={"size": 16})  # Increase font size here
        plt.title(f"Confusion Matrix for Class '{class_name}'", fontsize=18)  # Increase title font size
        plt.xlabel("Predicted", fontsize=16)  # Increase x-axis label font size
        plt.ylabel("Actual", fontsize=16)  # Increase y-axis label font size

    plt.tight_layout()
    plt.show()
    
# Updated evaluation function with class names
def evaluate_model_with_class_names(model, test_loader, device, class_names):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating Model", unit="batch"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            predictions = (output > 0.5).float()  # Threshold for multi-label classification

            y_true.append(target.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    overall_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    # Calculate per-label metrics using class names
    label_accuracies = (y_true == y_pred).mean(axis=0)
    label_precisions = precision_score(y_true, y_pred, average=None, zero_division=1)
    label_recalls = recall_score(y_true, y_pred, average=None, zero_division=1)
    label_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=1)

    # Print per-label metrics with class names
    print("\nPer-Label Metrics:")
    for i, (acc, prec, rec, f1) in enumerate(zip(label_accuracies, label_precisions, label_recalls, label_f1_scores)):
        class_name = class_names.get(i + 1, f"Label {i + 1}")
        print(f"{class_name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")

    return y_true, y_pred, label_accuracies, label_precisions, label_recalls, label_f1_scores



# In[14]:

# Visualize correct and incorrect predictions
def visualize_predictions(model, test_loader, device, n_images=5):
    correct = []
    incorrect = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Visualizing Predictions", unit="batch"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            predictions = (output > 0.5).float()

            for i in range(len(data)):
                if torch.equal(predictions[i], target[i]):
                    correct.append((data[i], predictions[i]))
                else:
                    incorrect.append((data[i], predictions[i], target[i]))

            if len(correct) >= n_images and len(incorrect) >= n_images:
                break

    # Plot correct predictions
    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for i in range(n_images):
        axes[i].imshow(correct[i][0].cpu().numpy().transpose((1, 2, 0)))
        axes[i].set_title(f"Pred: {correct[i][1].cpu().numpy()}")
        axes[i].axis('off')

    plt.show()

    # Plot incorrect predictions
    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for i in range(n_images):
        axes[i].imshow(incorrect[i][0].cpu().numpy().transpose((1, 2, 0)))
        axes[i].set_title(f"Pred: {incorrect[i][1].cpu().numpy()}\nTrue: {incorrect[i][2].cpu().numpy()}")
        axes[i].axis('off')

    plt.show()

# Evaluate the model on test data
#y_true, y_pred, label_accuracies, label_precisions, label_recalls, label_f1_scores = evaluate_model_with_per_label_metrics(model, test_loader, device)

class_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 
    90: 'toothbrush'
}


y_true, y_pred, label_accuracies, label_precisions, label_recalls, label_f1_scores = evaluate_model_with_class_names(
    model, test_loader, device, class_names
)

# Plot confusion matrices for the first 5 classes
plot_multilabel_confusion_matrices(y_true, y_pred, class_names, normalize=True)

# Visualize predictions
#visualize_predictions(model, test_loader, device, n_images=5)

