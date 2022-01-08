from torch import nn
import torch.nn.functional as fn
import numpy as np


class constants:
    """Constants and neural network hyperparameters"""
    DATASET_PATH = "D:\Datasets\PetImages" # Replace with the path to your dataset
    IMAGE_SIZE = 50 # Size of input images (50 x 50 pixels)
    LAYERS = 3 # Number of layers in the neural network
    EPOCHS = 5 # Number of times to loop over the entire dataset
    BATCH_SIZE = 16 # Number of images to be inputted at once (as a batch)
    TRAIN_SIZE = 0.8 # Percentage of the dataset to be used as training data
    TEST_SIZE = 0.2 # Percentage of the dataset to be used to test the model's accuracy
    KERNEL_SIZE = 3 # Size of the convolutional window scanning the image
    DROPOUT = 0.4 # Percentage of neurons to be disabled during a forward pass
    

class Model(nn.Module): 

    """Implementation of a convolutional neural network"""
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Model, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size)
        self.conv2_drop = nn.Dropout2d(self.dropout)
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = fn.relu(fn.max_pool2d(self.conv1(x), 2))
        x = fn.relu(fn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = fn.relu(self.fc1(x))
        x = fn.dropout(x, self.dropout)
        x = self.fc2(x)
        return x


def train(model, dataset, epochs, batch_size):
    batches = np.array_split(np.array(range(len(dataset))), batch_size)
    for epoch in range(epochs):
        for b in batches:
            pass

from . import process