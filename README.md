# Shape Classifier
## Overview
This project demonstrates the training of a Convolutional Neural Network (CNN) using PyTorch to classify images of basic geometric shapes (circle, square, triangle). The images are synthetically generated using Python's PIL library and stored in a custom dataset. The model is trained, validated, and tested on this dataset. The goal is to classify the images into one of the three shape categories.
## Requirements
```bash pip install torch torchvision matplotlib pillow scikit-learn ```

## Dataset
The dataset used in this project contains three types of geometric shapes: circles, squares, and triangles. These shapes are generated with random sizes and rotations. The dataset is split into three parts:
- Training (70%)
- Validation (20%)
- Testing (10%)

Images are saved under directories named after each shape: Circle, Square, and Triangle.

## Model
The architecture of the model is a simple CNN with 3 convolutional layers followed by fully connected layers:
- Conv1: 3 input channels → 16 output channels
- Conv2: 16 input channels → 32 output channels
- Conv3: 32 input channels → 64 output channels
- FC1: Flattened features → 256 neurons
- FC2: 256 neurons → Number of classes (3)

## Training
The model is trained using the following configurations:
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Cross-entropy loss
- Batch Size: 16
- Epochs: 30

The training loop includes periodic evaluation on the validation set, and the model is saved at regular intervals.

## Evaluation
After training, the model is evaluated on the test set, and the following metrics are calculated:
- **Test Accuracy**
- **Confusion Matrix** (to visualize model's performance across the classes)

## Results
After training for 30 epochs, the model achieved a 95.71% accuracy on the test set.

The confusion matrix and visualizations of predictions on test images will help in evaluating the model's performance.

## Usage
To run the Jupyter notebook:
1. Install Jupyter notebook:
```pip install notebook```

1. Launch Jupyter notebook:
```jupyter notebook```

1. Open the Shape_Classifier.ipynb file and execute the cells to train the model and visualize the results.
