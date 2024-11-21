import json
import random
import os

# Paths to my directories and annotation files
images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images"
annotations_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/annotations/annotations"
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Annotation files to process
annotation_files = [
    "captions", 
    "instances",   
    "person_keypoints"   
]

# Define split ratios (80:10:10)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def process_annotations(file_prefix):
    print(f"Processing {file_prefix} annotations...")
    
    # Paths to train and val annotation files
    train_file = os.path.join(annotations_dir, f"{file_prefix}_train2017.json")
    val_file = os.path.join(annotations_dir, f"{file_prefix}_val2017.json")
    print("the training files are",train_file)
    print("the validation files are",val_file)

    # Load the annotations
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(val_file, 'r') as f:
        val_data = json.load(f)

    # Combine train and val images and annotations
    combined_images = train_data["images"] + val_data["images"]
    combined_annotations = train_data["annotations"] + val_data["annotations"]

    # Shuffle combined images
    random.shuffle(combined_images)

    # Compute split sizes
    total_images = len(combined_images)
    train_size = int(total_images * train_ratio)
    print("train size is",train_size)
    val_size = int(total_images * val_ratio)
    print("validation size is",val_size)

    print("the length of total images is ",total_images)

# Process all annotation types
for file_prefix in annotation_files:
    process_annotations(file_prefix)


