import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.cuda.is_available())
sys.path.append(os.path.abspath('/home/rehan/Projects/Pytorch_Image_Classification/test_programs/task_2'))
# Import DataLoader objects for train and validation sets
from create_data_loaders import train_loader, val_loader

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Device selected: {device}")

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 90)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)  # Flatten the image tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Instantiate the model
model = MLPModel().to(device)
model.train()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 20
train_losses = []
val_losses = []

# Directory to save models
model_save_dir = './trained_models'
os.makedirs(model_save_dir, exist_ok=True)

# File to log losses
loss_log_file = os.path.join(model_save_dir, 'loss_log.txt')
with open(loss_log_file, 'w') as log_file:
    log_file.write("Epoch\tTraining Loss\tValidation Loss\n")

print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    # Training Loop
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
        # Show training progress at regular intervals
        if (batch_idx + 1) % 100 == 0:  # Adjust the number to print progress every X batches
            print(f"Batch {batch_idx + 1}/{len(train_loader)} - Training Loss: {loss.item():.4f}")
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
    
    # Validation Loop
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

    # Log losses to file
    with open(loss_log_file, 'a') as log_file:
        log_file.write(f"{epoch + 1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\n")

    # Save model for the current epoch
    model_save_path = os.path.join(model_save_dir, f'mlp_model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved for epoch {epoch + 1} at {model_save_path}")

print("Training completed.")

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid()
plt.savefig(os.path.join(model_save_dir, 'training_validation_loss.png'))
plt.show()
