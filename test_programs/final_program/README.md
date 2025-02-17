
# Multi-Label Classification on a Modified COCO Dataset
 The code includes:

- Data loading and transformation
- A custom `CocoDetection` class for dataset handling
- Splitting data into training, validation, and test sets
- Training loops for both a custom MLP model, CNN model and a fine-tuned ResNet50
- Evaluation with metrics such as **accuracy**, **precision**, **recall**, and **F1-score**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running the Code](#2-running-the-code)
  - [3. Understanding the Code Workflow](#3-understanding-the-code-workflow)
- [Customization](#customization)
- [Results and Evaluation](#results-and-evaluation)
- [License](#license)

---

## Prerequisites

- **Python 3.7+**
- **PyTorch** (with GPU support if available)
- **TorchVision**
- **NumPy**
- **Matplotlib**
- **PIL (Pillow)**
- **pandas**
- **seaborn**
- **scikit-learn**
- **pycocotools**
- **tqdm**
- **calflops**

Install the required packages using:

```bash
pip install -r requirements.txt
```

---


## Data Preparation

1. Place your modified COCO annotation file (example `instances_train2017_mod.json`) inside the `annotations/` folder. (Line 40)
2. Place your image data (COCO images or your own dataset) in the `train2017/` folder. (Line 145)
3. Adjust the paths in `final_program.py` if needed.

---

## Usage

### 1. Environment Setup

- Add in images and annotations inside the respective directories
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Ensure you are using a compatible Python version.

### 2. Running the Code

Run the script using:

```bash
python3 final_program.py
```

By default, the code will:

- Load the annotations file (`instances_train2017_mod.json`)
- Instantiate the `CocoDetection` dataset with appropriate transforms (resize to 224×224, normalization, etc.)
- Split the dataset into training, validation, and test sets
- Set up and train a multi-layer perceptron (MLP) model (`CustomMLP`) or a pretrained ResNet50 model or a CNN model
- Evaluate the model using metrics like accuracy, precision, recall, and F1-score
- Save model checkpoints during training

### 3. Understanding the Code Workflow

1. **Imports & Setup**  
   Loads required libraries, sets default parameters (e.g., figure size), and checks for GPU availability.

2. **Data Loading**  
   Reads the JSON annotation file and creates a `CocoDetection` dataset object. The dataset is split into training, validation, and test sets.

3. **Model Definition**  
   - `CustomMLP` is a simple MLP that flattens the input image and passes it through fully connected layers.
   - A CNN is also available for training
   - A pretrained ResNet50 model is also demonstrated for fine-tuning.

4. **Training & Evaluation**  
   Contains functions for training (`train_model`), validation (`validate_model`), and testing (`test_model`). The `train_and_evaluate` function coordinates the training and checkpointing process.

5. **Metrics**  
   After training, the code computes true positives, false positives, false negatives, and true negatives for each category. It then calculates accuracy, precision, recall, and F1-score.

6. **Logging & Checkpointing**  
   Training and validation losses are logged. Model weights are saved at each epoch (e.g., `model_state_epoch_xx.pt`). FLOPs, MACs, and Parameter are calculated for different models

---

## Customization

- **Changing the Model**  
  Replace or modify `CustomMLP` or the ResNet50 model or the CNN model as needed. Ensure the output layer matches the number of categories (80 for COCO).

- **Hyperparameters**  
  Adjust parameters such as `batch_size`, `learning_rate`, and `epochs` in the code.

- **Additional Metrics**  
  You can add further metrics (e.g., AUROC, mAP) in the evaluation section.

---

## Results and Evaluation

- **Loss Curves**  
  The code includes a section  that plots training and validation loss over epochs.

- **Confusion Matrix & Metrics**  
  The code prints true positives, false positives, etc., and computes accuracy, precision, recall, and F1-score per category.

```