import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os, sys, time, random


DATASET = os.path.expandvars("C:/Users/$USERNAME/Datasets/DogsVsCats")
IMAGE_SIZE = 50

def preprocess(dataset_path, mode="train"):
    base_path = os.path.dirname(dataset_path) 
    os.mkdir(os.path.join(base_path, f"arrays-{mode}"))
    for i, file in tqdm(enumerate(os.listdir(dataset_path))):
        image = Image.open(os.path.join(dataset_path, file)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.array(image).reshape(1, IMAGE_SIZE * IMAGE_SIZE)
        label = 0 if "cat" in file else 1
        output_path = os.path.join(base_path, f"arrays-{mode}", f"{str(i)}.npz")
        np.savez_compressed(output_path, x=array, y=label)

def load_batches(dataset_path, batch_size=16, mode="train"):
    path = os.path.join(dataset_path, f"arrays-{mode}")
    listdir = os.listdir(path)
    random.shuffle(listdir)
    for i in range(0, len(listdir), batch_size):
        file_batch = listdir[i:i+batch_size]
        if len(file_batch) != batch_size:
            continue
        x_batch, y_batch = [], []
        for file in file_batch:
            data = np.load(os.path.join(path, file))
            x_batch.append(data["x"][0])
            y_batch.append(data["y"])
        yield np.array(x_batch), np.array(y_batch)

train_loader = load_batches(DATASET, batch_size=16)
x, y = next(train_loader)

