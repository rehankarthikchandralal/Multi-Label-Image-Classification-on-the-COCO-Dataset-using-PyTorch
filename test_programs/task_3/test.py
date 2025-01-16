#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle  # For loading pre-saved dataloaders
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt  # For plotting losses

# Define the MLP model
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)  # First layer: 224x224x3 -> 512
        self.fc2 = nn.Linear(512, 512)           # Second layer: 512 -> 512
        self.fc3 = nn.Linear(512, 90)            # Output layer: 512 -> 90 (categories)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)  # Flatten input
        x = self.relu(self.fc1(x))    # Apply ReLU after fc1
        x = self.relu(self.fc2(x))    # Apply ReLU after fc2
        return self.sigmoid(self.fc3(x))  # Apply Sigmoid after fc3

# Function to validate the model
def validate_model(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Function to train the model
def train_model(model, device, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Progress bar for training
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} Training Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        # Validate model
        val_loss = validate_model(model, device, val_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{epochs} Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)

    return train_losses, val_losses

# Function to plot losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")  # Save the plot
    plt.show()

# Load dataloaders
with open('/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/train_loader.pkl', 'rb') as f:
    train_loader = pickle.load(f)

with open('/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/val_loader.pkl', 'rb') as f:
    val_loader = pickle.load(f)

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = CustomMLP().to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()  # Use Binary Cross-Entropy Loss for multi-label classification

# Train the model
train_losses, val_losses = train_model(mlp_model, device, train_loader, val_loader, optimizer, loss_fn, epochs=10)

# Save the model
torch.save(mlp_model.state_dict(), "mlp_model.pth")

# Plot the losses
plot_losses(train_losses, val_losses)
