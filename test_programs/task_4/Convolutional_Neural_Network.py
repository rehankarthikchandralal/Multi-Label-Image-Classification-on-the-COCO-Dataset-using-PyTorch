import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
from torchvision import transforms
from create_data_loaders import train_loader, val_loader  # Import DataLoader objects from task_2/create_data_loaders.py

class Convolutional_Neural_Network(nn.Module):
    def __init__(self):
        super(Convolutional_Neural_Network, self).__init__()
        
        # Convolutional Layer 1: 16 filters, 3x3 kernel, ReLU activation, padding=1, stride=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max-Pooling
        
        # Convolutional Layer 2: 32 filters, 3x3 kernel, ReLU activation, padding=1, stride=1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max-Pooling
        
        # Convolutional Layer 3: 64 filters, 3x3 kernel, ReLU activation, padding=1, stride=1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 Max-Pooling
        
        # Fully-Connected Layer: 80 outputs
        self.fc = nn.Linear(64 * 28 * 28, 80)  # 64 channels * 28x28 size after pooling
        
        # Output Layer: Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the network
        x = self.pool1(self.relu1(self.conv1(x)))  # Apply conv1 -> ReLU -> pool1
        x = self.pool2(self.relu2(self.conv2(x)))  # Apply conv2 -> ReLU -> pool2
        x = self.pool3(self.relu3(self.conv3(x)))  # Apply conv3 -> ReLU -> pool3
        
        # Flatten the output from convolutional layers to feed it into the fully connected layer
        x = x.view(-1, 64 * 28 * 28)  # Flatten the output
        
        # Fully connected layer
        x = self.fc(x)
        
        # Apply sigmoid activation to get the output
        x = self.sigmoid(x)
        
        return x


