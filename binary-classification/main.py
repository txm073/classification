import network
import numpy as np
import torch


net = network.Model(1, 10, 3, 0.4)
folder = "D:/Datasets/PetImages/arrays/"

print(net(batch))