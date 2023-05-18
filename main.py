import numpy as np
import pandas as pd
import tools.utils as gal_tools
import yaml
import datetime
import tools.feature_learning as fl
import Scripts.data_pipeline as dp
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
from Scripts import training


if __name__ == "__main__":
    training.train()
