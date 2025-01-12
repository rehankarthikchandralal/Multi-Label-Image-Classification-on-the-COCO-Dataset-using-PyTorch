import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
from tqdm import tqdm
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

# Add the correct path for your create_data_loaders module
sys.path.append(os.path.join(os.path.dirname(__file__), '../task_2'))

# Make sure to import COCOMultiLabelDataset from the correct module
from create_data_loaders import COCOMultiLabelDataset

print("Starting script execution...")

# Paths to image directories and annotations
train_img_dir = '/path/to/train/images'
val_img_dir = '/path/to/val/images'
train_ann_file = '/path/to/train/annotations.json'
val_ann_file = '/path/to/val/annotations.json'

# Define data transformations (example)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and Dataloaders for train and validation sets
train_dataset = COCOMultiLabelDataset(img_dir=train_img_dir, ann_file=train_ann_file, transform=data_transforms)
val_dataset = COCOMultiLabelDataset(img_dir=val_img_dir, ann_file=val_ann_file, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Check if DataLoaders are created successfully
if not train_loader or not val_loader:
    raise RuntimeError("Failed to create train_loader or val_loader.")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

# Convolutional Neural Network definition
class Convolutional_Neural_Network(nn.Module):
    def __init__(self):
        super(Convolutional_Neural_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 28 * 28, 80)  # Adjust for image size (you may need to change this based on your data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x  # CrossEntropyLoss expects raw logits, not softmax probabilities

# Flag to start fresh or resume training
start_fresh = True  # Set to False if you want to resume training

# Initialize the CNN model
model = Convolutional_Neural_Network().to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paths for saving and loading model, optimizer, and epoch
model_save_dir = '/home/rehan/Projects/Pytorch_Image_Classification/trained_model'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'cnn_model.pth')
optimizer_save_path = os.path.join(model_save_dir, 'optimizer.pth')
epoch_save_path = os.path.join(model_save_dir, 'epoch.txt')

# Load previously trained model, optimizer, and epoch if available and if not starting fresh
if not start_fresh and os.path.exists(model_save_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_save_path))
    if os.path.exists(optimizer_save_path):
        print("Loading saved optimizer state...")
        optimizer.load_state_dict(torch.load(optimizer_save_path))
    if os.path.exists(epoch_save_path):
        print("Loading last saved epoch...")
        with open(epoch_save_path, 'r') as f:
            start_epoch = int(f.read().strip())
        print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("Starting fresh training...")
    start_epoch = 0  # Start from the first epoch if starting fresh

# Set number of epochs
num_epochs = 10
train_losses = []
val_losses = []

# Print the number of batches for training and validation
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Start training process
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")

    # Training Loop
    model.train()
    running_train_loss = 0
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
        valid_mask = (labels >= 0) & (labels < 80)  # Filter out invalid labels
        images = images[valid_mask]
        labels = labels[valid_mask]

        if len(images) == 0 or len(labels) == 0:  # Skip invalid batches
            continue

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

    # Save model, optimizer, and epoch after each epoch
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    with open(epoch_save_path, 'w') as f:
        f.write(str(epoch + 1))

    # Validation Loop
    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(val_loader), desc="Validation"):
            valid_mask = (labels >= 0) & (labels < 80)
            images = images[valid_mask]
            labels = labels[valid_mask]

            if len(images) == 0 or len(labels) == 0:
                continue

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid()
plt.savefig("/home/rehan/Projects/Pytorch_Image_Classification/training_validation_loss.png")
plt.show()
