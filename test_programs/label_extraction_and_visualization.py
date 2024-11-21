import pandas as pd
import json
import matplotlib.pyplot as plt


# Mapping of category IDs to category names for the COCO dataset
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


def extract_labels_from_instances(json_file_path):
    # Load JSON data from the file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract annotations from the data
    annotations = data.get('annotations', [])
    
    # Extract category_ids from annotations
    category_ids = [annotation['category_id'] for annotation in annotations]
    
    # Remove duplicates by converting the list to a set
    unique_category_ids = set(category_ids)
    
    # Map category_id to category names using the mapping
    unique_labels = [category_mapping[cat_id] for cat_id in unique_category_ids if cat_id in category_mapping]
    
    return category_ids, unique_labels


json_file_path = '/home/rehan/Projects/Pytorch_Image_Classification/coco/annotations/annotations/instances_train2017.json'# Correct file path
category_ids, unique_labels = extract_labels_from_instances(json_file_path)

# Convert category_ids to a DataFrame to easily count occurrences
category_counts = pd.Series(category_ids).value_counts().sort_index()

# Print the number of different labels
print(f"Number of different labels: {len(unique_labels)}")

#  Visualize label distribution (histogram)
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar', color='green')
plt.title('Distribution of Object Categories in COCO Training Set')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(ticks=range(len(category_counts)), labels=category_counts.index.map(category_mapping), rotation=90)
plt.tight_layout()
plt.show()
#plt.savefig('/home/rehan/Projects/Pytorch_Image_Classification/doc/unique_labels.png') 