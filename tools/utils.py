import numpy as np
import PIL.Image as Image


def load_image(img_file):
    return np.asarray(Image.open(img_file))
