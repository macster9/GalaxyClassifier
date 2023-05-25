import numpy as np
import pandas as pd
import yaml
import os
import torch
import matplotlib.pyplot as plt
import _pickle as pickle
import PIL.Image as Image
import gc
from sklearn import metrics
from tqdm import tqdm


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


def plot(populated, actual, predicted, ax1, ax2, training_loss, validation_loss):
    conf_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(conf_matrix,
                                                display_labels=["SPIRAL", "ELLIPTICAL", "UNCERTAIN"])
    ax1.clear()
    cm_display.plot(ax=ax1, colorbar=False)
    ax1.title.set_text("Confusion Matrix")
    ax2.plot(training_loss, c="tab:blue", label="Training")
    ax2.plot(validation_loss, c="tab:orange", label="Validation")
    ax2.title.set_text("Loss Curve")
    plt.suptitle("Learning Metrics", fontsize=16)
    if not populated:
        ax2.legend()
    plt.pause(1)
    return None
