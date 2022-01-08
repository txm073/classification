import numpy as np # Array manipulation
from PIL import Image # Image manipulation
from tqdm import tqdm # Displays a progress bar during for loops
from network import constants
import os, sys, traceback

def _number_gen():
    """Infinitely yield a unique integer"""
    i = 0
    while True:
        yield i
        i += 1

def log(e, file):
    """Fetch tracebacks for any exceptions and append them to a file"""
    if not os.path.exists(file):
        with open(file, "w"): 
            pass
    with open(file, "a") as f:
        f.write("\n" + str(traceback.format_exc()))

def load_batch(batch_size):
    """Load a batch of data"""
    for i in range(batch_size, files):
        pass

def process_all(dataset_path):
    """
    Processes all of the files in the dataset:
      - Resizes them to the correct shape
      - Converts to greyscale to ensure there is only one colour channel
          so that we can use 2D convolutional layers instead of 3D
      - Attaches the correct label (0 or 1 corresponding to cat or dog)
      - Saves each image as an array file (.npy)
    """
    if not os.path.exists(dataset_path + "/arrays"):
        os.mkdir(dataset_path + "/arrays")
    gen = _number_gen()
    for label, image_type in enumerate(["Cat", "Dog"]):    
        print("Processing " + image_type + "s...")
        for item in tqdm(os.listdir(dataset_path + "/" + image_type)):
            try:
                path = dataset_path + "/" + image_type + "/" + item
                image = Image.open(path).convert("L")
                image = image.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE))
                try:
                    data = np.asarray(image, dtype=np.uint8)            
                except SystemError:
                    data = np.asarray(image.getdata(), dtype=np.uint8)
                save_path = dataset_path + "/arrays/" + str(next(gen)) + ".npz"
                np.savez_compressed(save_path, data=data, label=label)
            except (Exception, Warning) as e:
                log(e, "log.txt")