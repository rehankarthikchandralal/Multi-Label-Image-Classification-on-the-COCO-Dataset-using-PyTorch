import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, device, test_loader):
    """Evaluate the model on the test dataset and calculate metrics."""
    model.eval()  # Set the model to evaluation mode
    y_true = []  # True labels
    y_pred = []  # Predicted labels

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get model predictions
            output = model(data)
            predicted = (output > 0.5).float()  # Threshold at 0.5 for multi-label classification

            y_true.extend(target.cpu().numpy())  # Store true labels
            y_pred.extend(predicted.cpu().numpy())  # Store predictions

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true.flatten(), y_pred.flatten(), average='macro', zero_division=0)
    recall = recall_score(y_true.flatten(), y_pred.flatten(), average='macro', zero_division=0)
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='macro', zero_division=0)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and display confusion matrix for each class."""
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_predictions(model, device, test_loader, class_names, num_samples=5):
    """Visualize correct and incorrect predictions."""
    model.eval()
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = (output > 0.5).float()

            for i in range(len(data)):
                img = data[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                if torch.equal(predicted[i], target[i]):
                    correct_images.append((img, predicted[i].cpu().numpy()))
                else:
                    incorrect_images.append((img, predicted[i].cpu().numpy()))

                if len(correct_images) >= num_samples and len(incorrect_images) >= num_samples:
                    break

            if len(correct_images) >= num_samples and len(incorrect_images) >= num_samples:
                break

    print("Visualizing Correct Predictions:")
    for img, pred in correct_images[:num_samples]:
        plt.imshow(img)
        plt.title(f"Prediction: {pred}")
        plt.axis('off')
        plt.show()

    print("Visualizing Incorrect Predictions:")
    for img, pred in incorrect_images[:num_samples]:
        plt.imshow(img)
        plt.title(f"Prediction: {pred}")
        plt.axis('off')
        plt.show()

# Define class names for COCO dataset
class_names = list(catergory_id_to_name.values())

# Evaluate the model
y_true, y_pred = evaluate_model(oop_model, device, test_loader)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names)

# Visualize predictions
visualize_predictions(oop_model, device, test_loader, class_names)
