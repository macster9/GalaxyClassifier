import numpy as np
import pandas as pd
import yaml
import os
import torch
import matplotlib.pyplot as plt
import pickle
import PIL.Image as Image
import skimage.measure
from tqdm import tqdm


def query(obj_id):
    query = f"select"


def read_reference_table():
    with open('config.yml', 'r') as file:
        ref_table_dir = yaml.safe_load(file)["directories"]["ref_table"]
    ref_table = pd.read_csv(ref_table_dir)
    obj_id = ref_table["objid"]
    sample = ref_table["sample"]
    img_id = ref_table["asset_id"]
    return [obj_id, sample, img_id]


def read_galaxy_zoo_table():
    with open('config.yml', 'r') as file:
        gz_table_dir = yaml.safe_load(file)["directories"]["gal_zoo"]
    gz_table = pd.read_csv(gz_table_dir)
    obj_id = gz_table["OBJID"]
    ra = gz_table["RA"]
    dec = gz_table["DEC"]
    nvote = gz_table["NVOTE"]
    spiral = gz_table["SPIRAL"]
    elliptical = gz_table["ELLIPTICAL"]
    dk = gz_table["UNCERTAIN"]
    print(gz_table)
    return [obj_id, ra, dec, nvote, spiral, elliptical, dk]


def load_image(img_file):
      return np.asarray(Image.open(img_file))


def read_pickle(file):
    with open(file, "rb") as f:
        # data = np.array(pickle.load(f))
        data = pickle.load(f)
    return data


def save_images_to_pkl():
    with open('config.yml', 'r') as file:
        contents = yaml.safe_load(file)
        img_dir = contents["directories"]["images"]
        pickle_file_name = contents["directories"]["pickle_file_name"]

    img_list = []
    for img_file in tqdm(os.listdir(img_dir), desc="Transforming images to Tensor: "):
        if not img_file[-4:] == '.jpg':
            continue
        img_list.append(skimage.measure.block_reduce(load_image(os.path.join(img_dir, img_file)), (4, 4, 1), np.max))

    with open("data/images_tensor.pkl", "wb") as file:
        pickle.dump(torch.tensor(img_list), file, protocol=-1)
