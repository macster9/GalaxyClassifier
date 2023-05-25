import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tools import read, utils
import os


def load_image(img_file):
    return np.asarray(Image.open(img_file))


def image_analysis():
    contents = read.config()
    train_dir = contents["directories"]["train_dir"]
    img_list = os.listdir(train_dir)
    image = utils.load_image(os.path.join(train_dir, img_list[2]))
    utils.image_analysis(image)
    # new_image = np.reshape(image, (424*424, 3))
    # r, g, b = new_image.T
    # x = []
    # x.append(r)
    # x.append(g)
    # x.append(b)
    # print(np.std(x))
    # plt.hist(r, color="red")
    # plt.hist(g, color="green")
    # plt.hist(b, color="blue")
    # plt.show()
    image = np.where(image <= 3*np.std(image), 0, image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image)
    plt.show()
    return None
