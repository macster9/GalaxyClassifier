import os
import shutil
import yaml
from tools.read import config


def create_dirs():
    dirs = config()["directories"]
    dirs_to_mk = [
        "data", dirs["train_dir"], dirs["test_dir"], dirs["validate_dir"], dirs["model"]
    ]
    for dir in dirs_to_mk:
        if not os.path.exists(dir):
            os.mkdir(dir)
