import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
sys.path.append(os.path.abspath('/home/rehan/Projects/Pytorch_Image_Classification/test_programs/task_2'))
from create_data_loaders import train_loader, val_loader  # Import DataLoader objects

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Device selected: {device}")

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
        self.fc = nn.Linear(64 * 28 * 28, 80)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = Convolutional_Neural_Network().to(device)
model.train()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []
val_losses = []

# Directory to save models
model_save_dir = '/home/rehan/Projects/Pytorch_Image_Classification/trained_models'
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
    for batch_idx, (images, filenames) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = torch.zeros(images.size(0), 80).to(device)  # Dummy labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
    
    # Validation Loop
    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = torch.zeros(images.size(0), 80).to(device)  # Dummy labels
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
    model_save_path = os.path.join(model_save_dir, f'cnn_model_epoch_{epoch + 1}.pth')
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
plt.savefig("/home/rehan/Projects/Pytorch_Image_Classification/Training_and_Validation_Loss_Curves/training_validation_loss.png")
plt.show()