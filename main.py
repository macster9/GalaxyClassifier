import numpy as np
import pandas as pd
import tools.utils as galtools
import yaml
import tools.feature_learning as fl


if __name__ == "__main__":
    # galtools.save_images_to_pkl()
    with open("config.yml", "r") as file:
        file_name = yaml.safe_load(file)["directories"]["pickle_file_name"]
    data = galtools.read_pickle(file_name)
    print(len(data))
