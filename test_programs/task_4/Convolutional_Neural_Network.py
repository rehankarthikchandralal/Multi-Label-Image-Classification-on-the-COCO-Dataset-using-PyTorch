import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from create_data_loaders import train_loader, val_loader  # Import DataLoader objects

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Initialize the model and move it to the device (GPU/CPU)
model = Convolutional_Neural_Network().to(device)

# Set the model to training mode
model.train()

# Define a loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train
num_epochs = 5

# Training Loop
for epoch in range(num_epochs):
    running_loss = 0.0
    
    # Loop through the training data
    for batch_idx, (images, filenames) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
        
        # Move data to the appropriate device (GPU or CPU)
        images = images.to(device)  # Move images to GPU/CPU
        labels = torch.zeros(images.size(0), 80).to(device)  # Dummy labels (adjust based on your dataset)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item()
        
        # Print every 100 batches
        if batch_idx % 100 == 99:
            print(f"Batch {batch_idx + 1} - Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    
    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved as cnn_model.pth")
