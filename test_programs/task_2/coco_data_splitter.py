import json
import random
import os

"""
This script splits the COCO 2017 dataset annotations into train, validation, and test sets
based on a predefined ratio (80:10:10). It processes the annotations, filters them based on 
the image splits, and saves the new splits into separate JSON files.
"""

# Paths to directories and annotation files
images_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/images"
annotations_dir = "/home/rehan/Projects/Pytorch_Image_Classification/coco/annotations/annotations/"
output_dir = "/home/rehan/Projects/Pytorch_Image_Classification/split_datasets"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Define split ratios (80:10:10)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# File prefix to process
file_prefix = "instances"

# Define valid category mapping
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

def process_annotations(file_prefix):
    print(f"Processing {file_prefix} annotations...")

    # Path to the train annotation file
    train_file = os.path.join(annotations_dir, f"{file_prefix}_train2017.json")
    print("Loading the training file:", train_file)

    # Load the annotations
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    print("File loaded successfully. Processing data...")

    # Use only the train images and annotations
    combined_images = train_data["images"]
    combined_annotations = train_data["annotations"]

    print(f"Total images in dataset: {len(combined_images)}")
    print(f"Total annotations in dataset: {len(combined_annotations)}")

    # Filter out annotations that don't belong to valid categories
    print("Filtering annotations based on valid categories...")
    valid_annotations = [ann for ann in combined_annotations if ann["category_id"] in category_mapping]
    valid_image_ids = {ann["image_id"] for ann in valid_annotations}

    print(f"Valid annotations found: {len(valid_annotations)}")
    print(f"Images with valid annotations: {len(valid_image_ids)}")

    # Filter images to include only those with valid annotations
    valid_images = [img for img in combined_images if img["id"] in valid_image_ids]
    invalid_images = [img for img in combined_images if img["id"] not in valid_image_ids]

    print(f"Total valid images after filtering: {len(valid_images)}")
    print(f"Total invalid images (no valid annotations): {len(invalid_images)}")

    # Shuffle valid images
    print("Shuffling valid images...")
    random.shuffle(valid_images)

    # Compute split sizes
    total_images = len(valid_images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    print("Splitting dataset...")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {total_images - train_size - val_size}")

    # Split images into train, val, test
    train_images = valid_images[:train_size]
    val_images = valid_images[train_size:train_size + val_size]
    test_images = valid_images[train_size + val_size:]

    # Helper function to filter annotations based on image IDs
    def filter_annotations(images_split):
        image_ids = {img["id"] for img in images_split}
        return [ann for ann in valid_annotations if ann["image_id"] in image_ids]

    print("Filtering annotations for each split...")
    train_annotations = filter_annotations(train_images)
    val_annotations = filter_annotations(val_images)
    test_annotations = filter_annotations(test_images)

    print(f"Annotations - Train: {len(train_annotations)}, Validation: {len(val_annotations)}, Test: {len(test_annotations)}")

    # Save each split as a new JSON file
    def save_split(images_split, annotations_split, split_name):
        print(f"Saving {split_name} split...")
        split_data = {
            "images": images_split,
            "annotations": annotations_split
        }
        output_file = os.path.join(output_dir, f"{file_prefix}_{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(split_data, f)
        print(f"{split_name.capitalize()} split saved to {output_file}")

    save_split(train_images, train_annotations, "train")
    save_split(val_images, val_annotations, "val")
    save_split(test_images, test_annotations, "test")

    print(f"Completed splitting for {file_prefix} annotations!")
    print(f"Completed splitting for {file_prefix} annotations!")

# Process the "instances_train2017" annotation file only
process_annotations(file_prefix)
