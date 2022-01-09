import numpy as np
from PIL import Image
from tqdm import tqdm
import os, sys, time, random

import torch
from torch import nn, optim
from torch.nn import functional as func


DATASET = os.path.expandvars("C:/Users/$USERNAME/Datasets/DogsVsCats")
IMAGE_SIZE = 50
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-3

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
    load_batches.length = len(listdir)
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
        yield (
            torch.tensor(np.array(x_batch), dtype=torch.float32), 
            torch.tensor(np.array(y_batch), dtype=torch.float32)
        )


class Model(nn.Module):

    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, 1)
        for layer in [self.fc1, self.fc2, self.fc3, self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):    
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


train_loader = load_batches(DATASET, batch_size=BATCH_SIZE, mode="train")
test_loader = load_batches(DATASET, batch_size=BATCH_SIZE, mode="test")
model = Model(input_shape=IMAGE_SIZE * IMAGE_SIZE)
optimiser = optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()

def train():
    model.train()
    for i in range(EPOCHS):
        start = time.time()
        print(f"Starting epoch {i}...")
        j = 0
        while True:
            j += 1
            try:
                x_batch, y_batch = next(train_loader)
                output = model(x_batch)
                loss = loss_fn(output.reshape(BATCH_SIZE), y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                if j % 100 == 0:
                    print(f"Completed batch {j}")
            except StopIteration:
                break
        print(f"Completed epoch {i} in {time.time() - start}\n")
    torch.save(model.state_dict(), "model.pt")

train()
