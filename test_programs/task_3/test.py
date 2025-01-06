#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch  # PyTorch's core library for building and training deep learning models
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.optim as optim  # Import optimization algorithms such as Adam
import torch.nn.functional as F  # Import functional utilities like activation functions
from torch.utils.data import Dataset, DataLoader  # DataLoader for batching
import logging

# In[2]:

# Define the MLP Model
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()  # Call the parent class's constructor

        # First Layer
        self.fc1 = nn.Linear(224 * 224 * 3, 512)  # Fully connected layer that takes 224*224*3 Pixel input and maps it to 512 units.
        # Second Layer
        self.fc2 = nn.Linear(512, 512)  # Another fully connected layer that keeps the 512 units.
        # Output layer
        self.fc3 = nn.Linear(512, 90)  # Output layer that maps the 512 units to 90 output units (assuming 90 classes)

        # Define the activation function - ReLU (Rectified Linear Unit) - Sigmoid
        self.relu = nn.ReLU()  # ReLU introduces non-linearity after each layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid for multi-label classification

    # Define the forward pass (how data flows through the network)
    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)  # Flatten the input tensor from 224*224*3 to 150528

        x = F.relu(self.fc1(x))  # Pass data through the first layer and apply ReLU activation
        x = F.relu(self.fc2(x))  # Pass data through the second layer and apply ReLU activation

        return self.sigmoid(self.fc3(x))  # Pass data through the output layer and apply Sigmoid activation


# In[3]:

# Instantiate the MLP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = CustomMLP().to(device)  # Move the model to the device (GPU/CPU)

# In[4]:

# Define Loss function
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
# Define Optimizer
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)


# In[5]:

# Training function
def train_model(model, device, train_loader, optimizer, loss_fn):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize a variable to keep track of the cumulative loss for the epoch

    # Iterate over batches of data from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data and target to the appropriate device

        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing data through the model
        output = model(data)

        # Calculate the loss between the predicted outputs and the true labels
        loss = loss_fn(output, target)
        print("Loss:" + str(loss.item()))

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()

        # Update the model weights based on the computed gradients
        optimizer.step()

        running_loss += loss.item()  # Accumulate the loss for the current batch

    # Compute the average loss for the entire epoch
    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss  # Return the average training loss for this epoch


# In[6]:

# Validation function
def validate_model(model, device, val_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization)
    val_loss = 0.0  # Variable to accumulate validation loss

    # Iterate over batches of data from the validation set
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)  # Move data and target to the appropriate device

        # Forward pass: compute predicted outputs by passing data through the model
        output = model(data)

        # Calculate the loss between the predicted outputs and the true labels
        val_loss = loss_fn(output, target)
        print("Loss:" + str(val_loss.item()))

        val_loss += val_loss.item()  # Accumulate the loss for the current batch

    # Compute the average loss for the entire validation set
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss  # Return the average validation loss


# In[7]:

# Training and evaluation loop
def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    # Lists to store losses and accuracies
    train_losses = []  # To track training losses over epochs
    val_losses = []    # To track validation losses over epochs
    val_accuracies = []  # To track validation accuracies over epochs

    # Loop over the number of epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        logging.debug(f"\nEpoch {epoch + 1}/{epochs}")

        # Train the model and get training loss
        train_loss = train_model(model, device, train_loader, optimizer, loss_fn)  # Call the training function
        train_losses.append(train_loss)  # Store the training loss

        # Validate the model and get validation loss
        val_loss = validate_model(model, device, val_loader, loss_fn)  # Call the validation function
        val_losses.append(val_loss)  # Store the validation loss

        # Print training and validation results for the current epoch
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        logging.debug(f"Training Loss: {train_loss:.4f}")
        logging.debug(f"Validation Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"model_state_epoch_{epoch}.pt")

    return train_losses, val_losses


# In[8]:

# Example usage for training (replace train_loader and val_loader with actual loaders)
epochs = 5  # Set the number of epochs for training

logging.basicConfig(filename='train_validation_losses.log', level=logging.DEBUG)

# Track training and validation results
train_losses, val_losses = train_and_evaluate(
    mlp_model,  # The model to be trained and evaluated
    device,  # The device (CPU or GPU) where the model will run
    train_loader,  # DataLoader for training data
    val_loader,  # DataLoader for validation data
    optimizer,  # Optimizer to update model weights
    loss_fn,  # Loss function to compute the loss
    epochs=epochs  # Number of epochs to train for
)

# After training, you can analyze the recorded losses and accuracies
