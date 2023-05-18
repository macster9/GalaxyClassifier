import numpy as np
import pandas as pd
import yaml
import os
import torch
import matplotlib.pyplot as plt
import _pickle as pickle
import PIL.Image as Image
import gc
import skimage.measure
from tqdm import tqdm


def read_labels():
    with open(open_config()["directories"]["labels"], "r") as file:
        labels = pd.read_csv(file)
    return labels


def read_reference_table():
    ref_table_dir = open_config()["directories"]["ref_table"]
    ref_table = pd.read_csv(ref_table_dir)
    obj_id = np.array(ref_table["objid"])
    sample = np.array(ref_table["sample"])
    img_id = np.array(ref_table["asset_id"])
    return pd.DataFrame(np.array((sample, img_id)).T, columns=["SAMPLE", "IMG_ID"], index=obj_id)


def read_gz_table():
    gz_table_dir = open_config()["directories"]["gal_zoo"]
    gz_table = pd.read_csv(gz_table_dir)
    obj_id = gz_table["OBJID"]
    spiral = gz_table["SPIRAL"]
    elliptical = gz_table["ELLIPTICAL"]
    dk = gz_table["UNCERTAIN"]
    return pd.DataFrame(
        np.array((spiral, elliptical, dk)).T,
        columns=["SPIRAL", "ELLIPTICAL", "UNCERTAIN"],
        index=obj_id
    )


def load_image(img_file):
    return np.asarray(Image.open(img_file))


def read_pickle(file):
    gc.disable()
    with open(file, "rb") as f:
        data = pickle.load(f)
    gc.enable()
    return data


def open_config():
    with open("config.yml", "r") as file:
        contents = yaml.safe_load(file)
    return contents
