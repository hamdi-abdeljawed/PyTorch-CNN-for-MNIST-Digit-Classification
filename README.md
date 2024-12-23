# PyTorch-CNN-for-MNIST-Digit-Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The network is trained and tested on the MNIST dataset, and the trained model can predict digits from uploaded images.
Table of Contents

    Project Overview
    Setup & Installation
    Data Loading & Preprocessing
    CNN Model Architecture
    Training & Evaluation
    Model Prediction
    Results
    Conclusion
    License

Project Overview

This project uses a Convolutional Neural Network (CNN) to classify digits in the MNIST dataset. The model consists of two convolutional layers followed by fully connected layers, using ReLU activation and max pooling. The network is trained using the Adam optimizer and evaluated using the Cross-Entropy Loss function.

The model can be used to predict handwritten digits from user-uploaded images. The predicted digit is displayed alongside the uploaded image.
Setup & Installation

To run this project, you need Python 3.x and the following libraries:

    torch
    torchvision
    matplotlib
    numpy
    tqdm
    PIL (for image preprocessing)

You can install these libraries using pip:

pip install torch torchvision matplotlib numpy tqdm pillow

Data Loading & Preprocessing

The MNIST dataset is loaded from the torchvision.datasets module. The dataset consists of grayscale images (28x28 pixels) of handwritten digits (0-9). The following transformations are applied to the images:

    Convert images to tensors.
    Normalize the images to have a mean of 0.1307 and a standard deviation of 0.3081.

Training and test datasets are loaded using the DataLoader class, which also supports parallel data loading with multiple workers.
CNN Model Architecture

The CNN architecture consists of the following layers:

    Conv1 Layer: 32 filters, kernel size of 3x3, padding of 1.
    Conv2 Layer: 64 filters, kernel size of 3x3, padding of 1.
    MaxPool Layer: Pooling with kernel size of 2x2.
    Fully Connected Layers: Two fully connected layers with ReLU activations.
    Dropout: A dropout rate of 25% is applied after the first fully connected layer to prevent overfitting.

The final output layer has 10 neurons, corresponding to the 10 possible digits (0-9).
Training & Evaluation

The model is trained for 10 epochs using the Adam optimizer with a learning rate of 0.001. During training, both the loss and accuracy on the training and test datasets are logged.

The training process involves the following steps:

    Perform a forward pass through the network.
    Compute the loss using the cross-entropy loss function.
    Perform a backward pass and update the model parameters using the optimizer.

The model is evaluated on the test set after each epoch to track performance.
Model Prediction

The trained model can be used to predict handwritten digits from user-uploaded images. The following steps are performed for prediction:

    Upload an image.
    Preprocess the image by resizing it to 28x28 pixels and normalizing it.
    Pass the preprocessed image through the trained CNN model.
    Display the predicted digit along with the uploaded image.

Results

After training, the loss and accuracy on both the training and test sets are plotted to visualize the model's performance over epochs.

Example results:

    Training Loss: 0.1108, Training Accuracy: 97.65%
    Test Loss: 0.1076, Test Accuracy: 97.80%

Conclusion

This project demonstrates the use of a CNN for digit classification using the MNIST dataset. The model achieves high accuracy on both the training and test sets. The trained model can be used to classify digits in user-uploaded images.
