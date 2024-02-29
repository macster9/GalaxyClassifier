from tools import read, plot
from skimage.measure import block_reduce
from tqdm.auto import tqdm
import numpy as np
import os
import pandas as pd
import shutil


def run():
    labels_table = read.labels()
    plot.example_image(5, 5, labels_table)
    return None


def save_labels():
    ref_table = read.reference_table()
    gz_table = read.gz2_table2()
    meta_table = read.gz2_metadata_table()
    merged_tables = pd.merge(ref_table, gz_table, left_index=True, right_index=True)
    merged_meta = pd.merge(
        merged_tables, meta_table, left_index=True, right_index=True
    ).rename({0: "EXTINCTION"}, axis=1)
    merged_meta.to_csv("data/labels.csv")
    print("Saved Label Lookup Table.")
    return None


def split_datasets():
    # set random number seed
    np.random.seed(100)

    # get config info
    contents = read.config()
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


def process_image(image):
    image = np.where(image <= 3 * np.std(image), -1000, image)
    image = (image - np.min(abs(image))) / (np.max(image) - np.min(abs(image)))
    image = np.where(image <= 0, 0, image)
    image = block_reduce(image[108:315, 108:315], (4, 4, 1), np.max).T
    return image
