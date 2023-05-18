import torch
import pandas as pd
import tools.utils as gal_tools
from tools.feature_learning import CNN
from torchsummary import summary
import yaml
from skimage.measure import block_reduce
import numpy as np
import os
import matplotlib.pyplot as plt


def train():
    contents = gal_tools.open_config()
    train_dir, test_dir, valid_dir = [contents["directories"][x] for x in list(contents["directories"].keys())[-3:]]

    device = ["cuda:0" if torch.cuda.is_available() else "cpu"][0]
    model = CNN().to(device)
    summary(model, input_size=(3, 106, 106))
    labels_table = gal_tools.read_labels()

    for img in os.listdir(train_dir):
        obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
        label = torch.tensor(np.array(
            [int(obj["SPIRAL"]), int(obj["ELLIPTICAL"]), int(obj["UNCERTAIN"])]
        )).float().to(device)
        compressed_image = block_reduce(gal_tools.load_image(os.path.join(train_dir, img)), (4, 4, 1), np.max).T
        model_input = torch.tensor(compressed_image).unsqueeze(0).float().to(device)
        output = model(model_input)

        exit()
