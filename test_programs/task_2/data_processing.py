import json
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress bar

# Paths to directories and annotation files
input_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images/train2017"  # Path to images directory
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"      # Path where processed images will be saved
annotations_dir = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets"  # Path to annotations directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to Apply Resizing, Normalizing, and Augmenting All in One Step
def process_image(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Track the number of processed images, errors, and skipped images
processed_count = 0
error_count = 0
skipped_count = 0

# Load the list of image IDs from the instances JSON files (train, val, test)
def load_image_ids(split_file):
    with open(split_file, 'r') as f:
        data = json.load(f)
    return {img["id"] for img in data["images"]}

# Load image IDs from train, val, and test JSON files
train_image_ids = load_image_ids(os.path.join(annotations_dir, "instances_train.json"))
val_image_ids = load_image_ids(os.path.join(annotations_dir, "instances_val.json"))
test_image_ids = load_image_ids(os.path.join(annotations_dir, "instances_test.json"))

# Combine all the image IDs that should be processed
valid_image_ids = train_image_ids.union(val_image_ids, test_image_ids)

# Get the total number of images to process
total_images = len([image_name for image_name in os.listdir(input_dir) if image_name.lower().endswith('.jpg')])

# Iterate over all images in the directory with a progress bar
for image_name in tqdm(os.listdir(input_dir), total=total_images, desc="Processing Images"):
    image_path = os.path.join(input_dir, image_name)
    
    # Ensure it's a .jpg file
    _, extension = os.path.splitext(image_name)
    if extension.lower() == '.jpg':
        try:
            image_id = int(image_name.split('.')[0])  # Get the image ID by removing the extension and converting to int
            
            # Skip image if it is not part of the valid image IDs
            if image_id not in valid_image_ids:
                skipped_count += 1
                continue

            # Open the image
            image = Image.open(image_path)

            # Process the image (resize, normalize, augment)
            processed_image = process_image(image)

            # Save the processed image
            processed_image_pil = transforms.ToPILImage()(processed_image)
            processed_image_pil.save(os.path.join(output_dir, image_name))
            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error processing {image_name}: {e}")

# Final summary
print(f"\nProcessing complete. {processed_count} images processed successfully, {error_count} errors, {skipped_count} images skipped (not part of the train/val/test splits).")
