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

    # # Load the annotations
    # with open(train_file, 'r') as f:
    #     train_data = json.load(f)
    # with open(val_file, 'r') as f:
    #     val_data = json.load(f)

    # # Combine train and val images and annotations
    # combined_images = train_data["images"] + val_data["images"]
    # combined_annotations = train_data["annotations"] + val_data["annotations"]
    # categories = train_data["categories"]  # Categories remain the same

    # # Shuffle combined images
    # random.shuffle(combined_images)

    # # Compute split sizes
    # total_images = len(combined_images)
    # train_size = int(total_images * train_ratio)
    # val_size = int(total_images * val_ratio)

    # # Split images into train, val, test
    # train_images = combined_images[:train_size]
    # val_images = combined_images[train_size:train_size + val_size]
    # test_images = combined_images[train_size + val_size:]

#     # Helper function to filter annotations based on image IDs
#     def filter_annotations(images_split):
#         image_ids = {img["id"] for img in images_split}
#         return [ann for ann in combined_annotations if ann["image_id"] in image_ids]

#     # Filter annotations for each split
#     train_annotations = filter_annotations(train_images)
#     val_annotations = filter_annotations(val_images)
#     test_annotations = filter_annotations(test_images)

#     # Save each split as a new JSON file
#     def save_split(images_split, annotations_split, split_name):
#         split_data = {
#             "images": images_split,
#             "annotations": annotations_split,
#             "categories": categories
#         }
#         output_file = os.path.join(output_dir, f"{file_prefix}_{split_name}.json")
#         with open(output_file, 'w') as f:
#             json.dump(split_data, f)

#     save_split(train_images, train_annotations, "train")
#     save_split(val_images, val_annotations, "val")
#     save_split(test_images, test_annotations, "test")

#     print(f"Completed splitting for {file_prefix} annotations!")

# Process all annotation types
for file_prefix in annotation_files:
    process_annotations(file_prefix)

# print(f"All splits saved in {output_dir}")
