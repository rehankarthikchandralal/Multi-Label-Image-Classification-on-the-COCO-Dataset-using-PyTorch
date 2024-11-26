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


def process_annotations(file_prefix):

    """
    Processes the COCO annotations for a given file prefix (e.g., 'instances').
    It splits the data into train, validation, and test sets and saves them into 
    separate JSON files.

    Args:
        file_prefix (str): The prefix for the annotation file to process (e.g., 'instances').
    """

    print(f"Processing {file_prefix} annotations...")

    # Path to the train annotation file
    train_file = os.path.join(annotations_dir, f"{file_prefix}_train2017.json")
    print("The training file is:", train_file)

    # Load the annotations
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    # Use only the train images and annotations
    combined_images = train_data["images"]
    combined_annotations = train_data["annotations"]

    # Shuffle combined images
    random.shuffle(combined_images)

    # Compute split sizes
    total_images = len(combined_images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    print("Total images:", total_images)
    print("Train size:", train_size)
    print("Validation size:", val_size)

    # Split images into train, val, test
    train_images = combined_images[:train_size]
    val_images = combined_images[train_size:train_size + val_size]
    test_images = combined_images[train_size + val_size:]

    print("Train images:", len(train_images))
    print("Validation images:", len(val_images))
    print("Test images:", len(test_images))


    # Helper function to filter annotations based on image IDs
    def filter_annotations(images_split):

        """
        Filters the annotations based on the image IDs present in the provided image split.

        Args:
            images_split (list): A list of images to filter annotations by.

        Returns:
            list: A list of annotations corresponding to the provided image IDs.
        """

        image_ids = {img["id"] for img in images_split}
        return [ann for ann in combined_annotations if ann["image_id"] in image_ids]

    # Filter annotations for each split
    train_annotations = filter_annotations(train_images)
    val_annotations = filter_annotations(val_images)
    test_annotations = filter_annotations(test_images)

    print("Train annotations:", len(train_annotations))
    print("Validation annotations:", len(val_annotations))
    print("Test annotations:", len(test_annotations))


    # Save each split as a new JSON file
    def save_split(images_split, annotations_split, split_name):

        """
        Saves a specific image and annotation split (train, validation, or test) 
        into a JSON file.

        Args:
            images_split (list): List of images in the split.
            annotations_split (list): List of annotations for the images in the split.
            split_name (str): Name of the split (e.g., 'train', 'val', or 'test').
        """
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

# Process the "instances_train2017" annotation file only
process_annotations(file_prefix)

