import torch
from tools.utils import load_image
import numpy as np
import yaml
from tqdm import tqdm
import os
import pickle
import skimage.measure
import matplotlib.pyplot as plt


def save_images_to_pkl():
    with open('config.yml', 'r') as file:
        img_dir = yaml.safe_load(file)["directories"]["images"]

    img_list = []
    for img_file in tqdm(os.listdir(img_dir), desc="Transforming images to Tensor: "):
        if not img_file[-4:] == '.jpg':
            continue
        img_list.append(skimage.measure.block_reduce(
            load_image(os.path.join(img_dir, img_file)), (4, 4, 1), np.max
        ).shape)

    with open("data/images_tensor.pkl", "wb") as file:
        pickle.dump(torch.tensor(img_list), file, protocol=-1)



