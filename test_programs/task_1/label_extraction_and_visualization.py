import os
from PIL import Image
from torchvision import transforms
import torch
import json

# Directory paths
input_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"  # Path to images directory
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"      # Path where processed images will be saved
instances_test_file = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets/instances_test.json"  # Path to test annotations JSON file

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the instances_test.json to get valid image ids
with open(instances_test_file, 'r') as f:
    test_data = json.load(f)

# Create a set of valid image IDs from the annotations
valid_image_ids = set()
category_mapping = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 
    21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 
    26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 
    31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 
    40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 
    45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 
    50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 
    55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 
    60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 
    65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 
    70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 
    75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 
    80: "toothbrush"
}

# Populate the valid_image_ids set with the image IDs that have valid labels
for ann in test_data['annotations']:
    if ann['category_id'] in category_mapping:
        valid_image_ids.add(str(ann['image_id']).zfill(12))  # Make sure image_id matches the image file name format

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

# Track the number of processed images, errors, and invalid images
total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])  # Count all .jpg images
processed_count = 0
error_count = 0
invalid_count = 0  # Count of invalid images

# Process all images in the input directory
for idx, image_name in enumerate(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, image_name)
    
    # Check if the file is a .jpg image
    _, extension = os.path.splitext(image_name)
    if extension.lower() == '.jpg':  # Check if the extension is exactly '.jpg'
        try:
            # Check if the image is in the valid_image_ids set
            image_id = image_name.split('.')[0]
            if image_id not in valid_image_ids:
                invalid_count += 1
                continue  # Skip processing this image if it's not valid

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
        print(f"Processed {idx + 1}/{total_images} images... ({processed_count} successfully processed, {error_count} errors, {invalid_count} invalid images)")

# Final summary
print(f"\nProcessing complete. {processed_count} images processed successfully, {error_count} errors, {invalid_count} invalid images skipped.")
