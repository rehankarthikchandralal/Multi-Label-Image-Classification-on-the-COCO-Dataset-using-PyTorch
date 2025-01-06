import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../task_2'))
# Make sure to import ProcessedImagesDataset from the correct module
from create_data_loaders import ProcessedImagesDataset
from create_data_loaders import custom_collate_fn

print("Starting script execution...")

# Use the absolute path to the pickle files as previously defined
train_loader_path = '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/train_loader.pkl'
val_loader_path = '/home/rehan/Projects/Pytorch_Image_Classification/dataloaders/val_loader.pkl'

train_loader = None
val_loader = None

if os.path.exists(train_loader_path) and os.path.exists(val_loader_path):
    print("Loading saved DataLoader objects from pickle files...")
    try:
        with open(train_loader_path, 'rb') as f:
            train_loader = pickle.load(f)
        with open(val_loader_path, 'rb') as f:
            val_loader = pickle.load(f)
        print("DataLoader objects loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load DataLoader objects: {str(e)}")
else:
    print("Saved DataLoader files not found. Proceeding with dataset creation...")

# Check if the DataLoader objects are loaded successfully
if not train_loader or not val_loader:
    raise RuntimeError("Failed to load train_loader or val_loader.")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

num_epochs = 5
train_losses = []
val_losses = []

print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    # Training Loop
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
        if images is None or labels is None:  # Skip invalid data
            continue

        images = images.to(device)
        labels = labels.to(device)  # Get real labels from DataLoader
        
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
        for batch_idx, (images, labels) in tqdm(enumerate(val_loader), desc="Validation"):
            if images is None or labels is None:  # Skip invalid data
                continue

            images = images.to(device)
            labels = labels.to(device)  # Get real labels from DataLoader
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

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
plt.savefig("/home/rehan/Projects/Pytorch_Image_Classification/training_validation_loss.png")
plt.show()

# Save the model
model_save_dir = '/home/rehan/Projects/Pytorch_Image_Classification/trained_model'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'cnn_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved as {model_save_path}")
