import os
from PIL import Image
from torchvision import transforms
import torch

# Directory paths
input_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"  # Path to images directory
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"      # Path where processed images will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to Apply Resizing, Normalizing, and Augmenting All in One Step
def process_image(image):
    # Define the transformation pipeline (resize, normalize, augment)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),    # Random crop to 224x224
        transforms.RandomHorizontalFlip(),    # Random horizontal flip
        transforms.RandomRotation(30),        # Random rotation between -30 and 30 degrees
        transforms.ToTensor(),                # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
    ])
    
    # Apply transformations to the image
    return transform(image)

# Track the number of processed images and errors
total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])  # Count all .jpg images
processed_count = 0
error_count = 0

# Process all images in the input directory
for idx, image_name in enumerate(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, image_name)
    
    # Check if the file is a .jpg image
    _, extension = os.path.splitext(image_name)
    if extension.lower() == '.jpg':  # Check if the extension is exactly '.jpg'
        try:
            # Open the image
            image = Image.open(image_path)

            # Process the image with resizing, normalizing, and augmenting
            processed_image = process_image(image)

            # Convert the processed tensor back to a PIL image for saving
            processed_image_pil = transforms.ToPILImage()(processed_image)

            # Save the processed image with its original name in the output directory
            processed_image_pil.save(os.path.join(output_dir, image_name))
            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error processing {image_name}: {e}")

    # Print progress every 100 images
    if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
        print(f"Processed {idx + 1}/{total_images} images... ({processed_count} successfully processed, {error_count} errors)")

# Final summary
print(f"\nProcessing complete. {processed_count} images processed successfully, {error_count} errors occurred.")
