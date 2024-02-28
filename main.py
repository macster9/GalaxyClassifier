import numpy as np
import pandas as pd
import tools.utils as gal_tools
import yaml
import datetime
import tools.network as fl
import scripts.data_pipeline as dp
from tools import read, utils
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
from scripts import training


if __name__ == "__main__":
    # training.train(learning_rate=0.0001, epochs=3)
    # dp.run()
    img = os.listdir(read.config()["directories"]["train_dir"])[0]
    labels_table = read.labels()
    obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
    print(obj)
