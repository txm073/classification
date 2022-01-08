import os
import sys
import shutil

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Produces batch of data:
# - train_x: numpy array of the image itself
# - train_y: one-hot encoded vector representing the label (hopefully equal to the NN output)
def get_batch(batch_size=10, dtype="train"):
    data_x, data_y = [], []
    for i in range(batch_size):
        if dtype == "train":
            image = next(images)

        elif dtype == "test":
            image = next(testdata)

        image_data = pad_image(to_array(image[0]))
        label = OHEncode(int(image[1]))
        data_x.append(image_data)
        data_y.append(label)

    return data_x, data_y

# Loops through every categorised image in the training set
def _get_image():
    for _class in os.listdir(dataset_dir):
        for image in os.listdir(os.path.join(dataset_dir, _class, "train")):
            path = os.path.join(dataset_dir, _class, "train", image)
            yield path, _class

# One hot encodes the image label for the 'train_y' output data
def OHEncode(num):
    array = [0] * 10
    array[num] = 1
    return array

def dataset_size(dtype="train"):
    class_lengths = [len(os.listdir(os.path.join(dataset_dir, _class, dtype)))
                        for _class in os.listdir(dataset_dir)]

    return sum(class_lengths)

# Loops through and yields each image from the test data
def _get_test_data():
    for _class in os.listdir(dataset_dir):
        for image in os.listdir(os.path.join(dataset_dir, _class, "test")):
            path = os.path.join(dataset_dir, _class, "test", image)
            yield path, _class

# Displays an image with matplotlib
def display(array):
    plt.imshow(array)
    plt.show()

# Generates a unique number for each image in the dataset
def numbers():
    for i in range(1000000000):
        yield i

# Convert image to numpy array
def to_array(file):
    return cv2.imread(file)

# Image must be padded before it can be inputted into the networks
def pad_image(array):
    target_dims = 100
    if not isinstance(array, np.ndarray):
        array = to_array(array)

    h, w, _ = array.shape
    padx, pady = target_dims - w, target_dims - h
    array = np.pad(array, ((0, pady), (0, padx), (0, 0)))
    return array

# Train the network
def train(epochs=3, batch_size=10):
    net = Net()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.0001)


    batches_per_epoch = dataset_size() // batch_size
    for epoch in tqdm(range(epochs)):
        print(f"\nStarting epoch {epoch}...")
        batch_count = 0
        for batch in range(batches_per_epoch + 1):
            try:
                train_x, train_y = get_batch(batch_size=batch_size)
                batch_count += 1
                print(f"Forwarding batch {batch_count} into the network...")
            except StopIteration:
                break
        print(f"\nEpoch {epoch} now complete!")

# Ensure that there is similar amounts of data for each class
# Otherwise the model will make inaccurate predictions
def balance_dataset():
    class_nums = [len(os.listdir(os.path.join(dataset_dir, _class, "train")))
                    for _class in os.listdir(dataset_dir)]

    total = sum(class_nums)
    print("Data percentages: ")
    for i, amount in enumerate(class_nums):
        print("Class:", i, "-", str(amount / total * 100)+"%")

    max_cutoff = max(class_nums) // 4
    if max(class_nums) - min(class_nums) > max_cutoff:
        print("Class delta is too large")
        return None

    minimum = min(class_nums)
    for index, _class in enumerate(os.listdir(dataset_dir)):
        for file in os.listdir(os.path.join(dataset_dir, _class))[:class_nums[index] - minimum]:
            #os.remove(file)
            pass

# Only called once
# Used to collect all the images and split them into testing and training
# Also renames all the files
def organise_dataset(path):
    current_dir = os.getcwd()
    try:
        os.mkdir(os.path.join(current_dir, "dataset"))
        for i in range(10):
            os.mkdir(os.path.join(current_dir, "dataset", str(i)))
            os.mkdir(os.path.join(current_dir, "dataset", str(i), "train"))
            os.mkdir(os.path.join(current_dir, "dataset", str(i), "test"))
    except FileExistsError:
        pass

    number = numbers()
    # Loop through base directories
    for root_dir in os.listdir(path)[:-1]:
        # Loop through digit classes
        print("Root directory:", root_dir)
        for _class in os.listdir(os.path.join(path, root_dir)):
            print("Number class:", _class)
            if os.path.splitext(os.path.join(path, root_dir, _class))[1] == "":
                # Loop through image files within each digit
                for index, image in enumerate(os.listdir(os.path.join(path, root_dir, _class))):
                    source = os.path.join(path, root_dir, _class, image)
                    new_name = os.path.join(path, root_dir, _class, str(next(number))+".png")
                    os.rename(source, new_name)
                    # There are 60 files in each sub folder (from the GitHub repository)
                    # 5/6 of the images go in the training dataset, the rest go in the testing set
                    if index <= 50:
                        sub_dest = "train"
                    else:
                        sub_dest = "test"

                    destination = os.path.join(current_dir, "dataset", str(_class), sub_dest)
                    try:
                        shutil.move(new_name, destination)
                    except shutil.Error:
                        pass

images = _get_image()
testdata = _get_test_data()
dataset_dir = os.path.join(os.path.dirname(sys.argv[0]), "dataset")

if __name__ == "__main__":
    # Set the data generators as global variables
    # This ensures that the generator objects are not re-instantiated when the batch function is called
    # Otherwise the batch function would always return the first batch of data
    images = _get_image()
    testdata = _get_test_data()
    dataset_dir = os.path.join(os.path.dirname(sys.argv[0]), "dataset")
    #os.chdir(os.path.dirname(sys.argv[0]))
    input("Are you sure you want to continue?")
    train_x, train_y = get_batch(1)
    print(train_x[0].shape)