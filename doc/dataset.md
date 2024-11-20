The annotations folder contains several files divided into training and validation sets
captions give general description for the images
Instances files contain annotations for object detection, where objects within the images are identified and labeled
bounding boxes and ifcrowd flag is used to simplify the data for supervised learning, instead of just a label associated with an image, we also give additional information that will help us train the model well to do classification

Person files contain annotations for human pose estimation. Specifically, they describe the locations of key body joints (keypoints) for people in Instances 
