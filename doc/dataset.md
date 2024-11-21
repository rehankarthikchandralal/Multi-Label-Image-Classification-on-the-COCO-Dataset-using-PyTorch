The annotations folder contains several files divided into training and validation sets
captions give general description for the images
Instances files contain annotations for object detection, where objects within the images are identified and labeled
bounding boxes and ifcrowd flag is used to simplify the data for supervised learning, instead of just a label associated with an image, we also give additional information that will help us train the model well to do classification

Person files contain annotations for human pose estimation. Specifically, they describe the locations of key body joints (keypoints) for people in Instances 
The COCO dataset is a multi-label dataset, where each image can contain multiple objects, and therefore, multiple labels (categories) are associated with a single image
There is a category id for the annotations in instance file
all content in person_keypoints files have category id of 1 as it represents a person
Disadvantage of label imbalance in Multilabel classifier
The model may focus on learning patterns for the majority labels, leading to a lack of sensitivity to minority labels.
The loss function may be dominated by the contribution from majority labels, making it difficult for the model to learn from the minority labels.
   Keep the label imbalance problem
Yes, all 6 of the files you mentioned represent annotations in different formats or types of annotation for the COCO dataset. 
Resizing images to a consistent size, like 224x224 pixels, is a common practice in machine learning, particularly when working with convolutional neural networks (CNNs).

CNNs expect input data of a fixed size. Resizing ensures that all images have the same dimensions, making them compatible with the network's architecture
Smaller images require less computational resources, leading to faster training and inference times.

 Images and annotations are saved together in JSON format after coco_data_splitter.py
 Reason:By saving images and annotations in JSON format, you create a flexible, readable, and interoperable representation of your data. 
 2 images are lost , i think because of int explicit conversion