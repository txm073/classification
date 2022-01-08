import cv2
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
import matplotlib.pyplot as plt
import os


def process(path):
    for digit in os.listdir(path):
        print(f"Processing class {int(digit) + 1}...")
        train_path = os.path.join(path, digit, "train")
        test_path = os.path.join(path, digit, "test")
        save_train_path = os.path.join(os.path.dirname(path), "arrays", "digit", "train")
        save_test_path = os.path.join(os.path.dirname(path), "arrays", "digit", "test")

        for file in tqdm(os.listdir(train_path)):
            arr = cv2.imread(os.path.join(train_path, file))
            arr = arr[:, :, 0]
            if arr.shape[0] > 100:
                arr = arr[:100, :, :]
            elif arr.shape[1] > 100:
                arr = arr[:, :100, :]
            print(arr.shape)
            for row in arr:
                print(row)
            plt.figure()
            plt.imshow(arr)
            plt.show()
            return

process(os.path.join(os.path.dirname(__file__), "dataset"))
