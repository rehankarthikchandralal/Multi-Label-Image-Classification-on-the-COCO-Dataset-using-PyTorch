#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import logging
import os
import sys
from tqdm import tqdm  # For the progress bar
import csv  # For saving losses in CSV format
import matplotlib.pyplot as plt  # For plotting loss curves
# Add the correct path for your create_data_loaders module
sys.path.append(os.path.join(os.path.dirname(__file__), '../task_2'))

from create_data_loaders import ProcessedImagesDataset
from create_data_loaders import custom_collate_fn

print("Starting script execution...")

train_loader_path = '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/train_loader.pkl'
val_loader_path = '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/val_loader.pkl'

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()

        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 90)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = CustomMLP().to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

checkpoint_path = "checkpoint.pth"  # Path to save/load checkpoint

def save_checkpoint(model, optimizer, epoch, loss, filename=checkpoint_path):
    """Save model and optimizer state, and current epoch"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filename=checkpoint_path):
    """Load model and optimizer state, and resume training from saved checkpoint"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model = CustomMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # We want to resume from the next epoch
        print(f"Resuming from epoch {start_epoch} with loss {checkpoint['loss']:.4f}")
        return model, optimizer, start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        model = CustomMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer, 0  # Start from epoch 0

def load_dataloaders(train_loader_path, val_loader_path):
    with open(train_loader_path, 'rb') as f_train, open(val_loader_path, 'rb') as f_val:
        train_loader = pickle.load(f_train)
        val_loader = pickle.load(f_val)
    return train_loader, val_loader


def filter_invalid_labels(data, target):
    valid_mask = (target >= 0) & (target < 80)  # Boolean mask for valid labels
    data = data[valid_mask]  # Filter images
    target = target[valid_mask]  # Filter labels
    return data, target

def train_model(model, device, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        data, target = data.to(device), target.to(device)

        data, target = filter_invalid_labels(data, target)

        if target.dim() == 1:  # Target is a 1D vector of labels, not one-hot encoded
            target = F.one_hot(target, num_classes=90).float()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss


def validate_model(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        data, target = data.to(device), target.to(device)

        data, target = filter_invalid_labels(data, target)

        if target.dim() == 1:  # Target is a 1D vector of labels, not one-hot encoded
            target = F.one_hot(target, num_classes=90).float()

        output = model(data)
        loss = loss_fn(output, target)

        val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, epochs=5, save_interval=3):
    train_losses = []
    val_losses = []

    # Open the text file to log the losses for each epoch
    with open('epoch_losses.txt', 'a') as loss_file, open('train_validation_losses.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()

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

            # Save the model's state after every 'save_interval' epochs
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(model, optimizer, epoch, val_loss)

            # Write the losses to the CSV file
            writer.writerow({'Epoch': epoch + 1, 'Training Loss': train_loss, 'Validation Loss': val_loss})

            # Write the losses to the text file
            loss_file.write(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

    return train_losses, val_losses




def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()


epochs = 3
save_interval = 1  # Save checkpoint every 1 epoch
logging.basicConfig(filename='train_validation_losses.log', level=logging.DEBUG)

# Load dataloaders
train_loader, val_loader = load_dataloaders(
    '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/train_loader.pkl',
    '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/val_loader.pkl'
)

# Load checkpoint or start from scratch
mlp_model, optimizer, start_epoch = load_checkpoint()

# Start training from the last saved epoch or from epoch 0
train_losses, val_losses = train_and_evaluate(
    mlp_model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs=epochs,
    save_interval=save_interval
)

# After training, plot the loss curves
plot_loss_curve(train_losses, val_losses)
