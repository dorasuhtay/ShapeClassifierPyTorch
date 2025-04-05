# Shape Classifier
## Overview
This project demonstrates the training of a Convolutional Neural Network (CNN) using PyTorch to classify images of basic geometric shapes (circle, square, triangle). The images are synthetically generated using Python's PIL library and stored in a custom dataset. The model is trained, validated, and tested on this dataset. The goal is to classify the images into one of the three shape categories.
## Requirements
pip install torch torchvision matplotlib pillow scikit-learn

## Dataset
The dataset used in this project contains three types of geometric shapes: circles, squares, and triangles. These shapes are generated with random sizes and rotations. The dataset is split into three parts:
- Training (70%)
- Validation (20%)
- Testing (10%)

Images are saved under directories named after each shape: Circle, Square, and Triangle.

## Model
The model used in this project is a simple Convolutional Neural Network (CNN). It has the following architecture:
1. Conv Layer 1: 3 input channels (RGB), 16 output channels, kernel size 3, padding 1.
1. Conv Layer 2: 16 input channels, 32 output channels, kernel size 3, padding 1
1. MaxPool Layer: 2x2 max pooling
1. Fully Connected Layer 1: 32 * 16 * 16 inputs, 128 outputs
1. Fully Connected Layer 2: 128 inputs, 3 outputs (for 3 classes)

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
The model's performance is evaluated and displayed via:
- Confusion Matrix
- Sample predictions with their actual labels
