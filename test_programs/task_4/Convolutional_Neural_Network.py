import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import time

# Paths
data_loader_save_dir = "/home/rehan/Projects/Pytorch_Image_Classification/dataloaders"

print("Starting script execution...")

# Step 1: Load DataLoaders
print("Loading DataLoaders...")
start_time = time.time()
try:
    with open(os.path.join(data_loader_save_dir, "train_loader.pkl"), "rb") as f:
        train_loader = pickle.load(f)
    print("Train DataLoader loaded successfully.")

    with open(os.path.join(data_loader_save_dir, "val_loader.pkl"), "rb") as f:
        val_loader = pickle.load(f)
    print("Validation DataLoader loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load DataLoader objects: {str(e)}")
print(f"DataLoaders loaded in {time.time() - start_time:.2f} seconds.")

# Step 2: Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

# Step 3: Define CNN model
print("Initializing the model...")
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
print("Model initialized successfully.")

# Step 4: Set up loss function and optimizer
print("Setting up loss function and optimizer...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Loss function and optimizer set.")

# Step 5: Training and validation
num_epochs = 5
train_losses = []
val_losses = []

print("Starting training process...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs} starting...")
    epoch_start_time = time.time()
    running_train_loss = 0.0
    running_val_loss = 0.0

    # Training loop
    print("Training...")
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training Progress"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    print("Validating...")
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation Progress"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {time.time() - epoch_start_time:.2f} seconds.")

print("Training process completed.")

# Step 6: Plot the training and validation loss curves
print("Plotting training and validation loss curves...")
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid()
plot_save_path = "/home/rehan/Projects/Pytorch_Image_Classification/training_validation_loss.png"
plt.savefig(plot_save_path)
plt.show()
print(f"Loss curves saved as {plot_save_path}")

# Step 7: Save the model
print("Saving the trained model...")
model_save_dir = '/home/rehan/Projects/Pytorch_Image_Classification/trained_model'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'cnn_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved successfully at {model_save_path}")

print("Script execution completed.")
