import tools.utils as gal_tools
import yaml
import torch
import skimage.measure
from tqdm import tqdm
import numpy as np
import pickle
import os
import pandas as pd
import shutil


def save_labels():
    ref_table = gal_tools.read_reference_table()
    gz_table = gal_tools.read_gz_table()
    merged_table = pd.merge(ref_table, gz_table, left_index=True, right_index=True)
    merged_table.to_csv("data/labels.csv")
    return None


def split_datasets():
    # set random number seed
    np.random.seed(100)

    # get config info
    contents = gal_tools.open_config()
    image_dir, train_dir, test_dir, valid_dir = [
        contents["directories"][x] for x in list(contents["directories"].keys())[-4:]
    ]
    train_split, test_split = [contents["data"][x] for x in list(contents["data"].keys())]

    # check directories are empty, if they exist
    if os.path.exists(train_dir):
        assert len(os.listdir(train_dir)) == 0, "Training Directory not empty."
    elif os.path.exists(test_dir):
        assert len(os.listdir(test_dir)) == 0, "Test Directory not empty."
    elif os.path.exists(valid_dir):
        assert len(os.listdir(valid_dir)) == 0, "Validation Directory not empty."
    else:
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(valid_dir)

    # get list of images and randomise
    img_list = [ele for ele in os.listdir(image_dir) if ".jpg" in ele]
    np.random.shuffle(img_list)

    # iterate over list of images, moving them into different folders
    for img in tqdm(img_list, desc="Splitting Images to train/test/validate: "):
        prob = np.random.random()
        if prob < train_split:
            shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, img))
        elif (prob >= train_split) & (prob < train_split+test_split):
            shutil.move(os.path.join(image_dir, img), os.path.join(test_dir, img))
        else:
            shutil.move(os.path.join(image_dir, img), os.path.join(valid_dir, img))
    print("Dataset Split Complete.")
    return None
