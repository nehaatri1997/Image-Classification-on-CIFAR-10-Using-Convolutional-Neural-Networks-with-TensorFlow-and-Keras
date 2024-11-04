# Image-Classification-on-CIFAR-10-Using-Convolutional-Neural-Networks-with-TensorFlow-and-Keras

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. CIFAR-10 is a dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project includes data preprocessing, model building, training with data augmentation, evaluation, and visualization.

# Requirements
To run this project, you need Python installed with the following libraries:
tensorflow
keras (included within TensorFlow)
numpy
matplotlib
scikit-learn

# Dataset
The CIFAR-10 dataset is automatically downloaded through Keras. It consists of 10 classes:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
Each image is a 32x32 RGB image.

# Project Overview
Data Loading and Preprocessing: The CIFAR-10 dataset is loaded and split into training and test sets. The pixel values are normalized to a range of 0 to 1 to facilitate better model training.

# Model Architecture:

The model is a Convolutional Neural Network (CNN) with three convolutional layers, followed by max-pooling layers.
A flattening layer and two dense layers are added, with the final layer containing 10 units (one for each class) with softmax activation for multiclass classification.

# Data Augmentation: 

An ImageDataGenerator is used for on-the-fly data augmentation, including random rotations, width/height shifts, and horizontal flips to improve model generalization.
Model Training and Evaluation: The model is compiled and trained on the augmented data for 10 epochs. The accuracy and loss are evaluated on the test set and displayed.

Visualization: Training and validation accuracy are plotted over epochs to analyze the model's performance during training.

# Usage

Run the Code: Save the script as a .py file and execute it. The script will load the CIFAR-10 dataset, preprocess the data, build and train the CNN model, and display results.

# Customize Parameters:

Adjust parameters in ImageDataGenerator for different data augmentation techniques.
Modify the number of epochs, batch size, or model layers to experiment with performance.

# Results

The model achieves a certain level of accuracy on the CIFAR-10 test dataset, which can be improved with further tuning or additional data augmentation.

# Notes
Data Augmentation: Experiment with different augmentation techniques to see their impact on model accuracy.
Hyperparameters: The learning rate, batch size, and model architecture can be adjusted for potentially better results.
