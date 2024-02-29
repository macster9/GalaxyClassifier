import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tools import read, utils


def learning_metrics(populated, actual, predicted, ax1, ax2, training_loss, validation_loss):
    conf_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(conf_matrix,
                                                display_labels=["SPIRAL", "ELLIPTICAL", "UNCERTAIN"])
    ax1.clear()
    cm_display.plot(ax=ax1, colorbar=False)
    ax1.title.set_text("Confusion Matrix")
    ax2.title.set_text("Loss Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    plt.suptitle("Learning Metrics", fontsize=16)
    if not populated:
        ax2.plot(training_loss, c="tab:blue", label="Training")
        ax2.plot(validation_loss, c="tab:orange", label="Validation")
        ax2.legend()
    else:
        ax2.plot(training_loss, c="tab:blue")
        ax2.plot(validation_loss, c="tab:orange")
    plt.pause(1)
    return None


def example_image(x, y, lookup_table):
    img_dir = read.config()["directories"]["test_dir"]
    fig, axs = plt.subplots(x, y, figsize=(10, 10))
    img_list = os.listdir(img_dir)
    # np.random.shuffle(img_list)
    for i in range(y):
        for j in range(x):
            rand_image = np.random.choice(img_list)
            obj = lookup_table.loc[lookup_table["IMG_ID"] == int(rand_image[:-4])]
            for name in obj.columns:
                if isinstance(obj[name], int):
                    if obj[name].item() == 1:
                        axs[i, j].set_title(name, fontsize=10)
            image = utils.load_image(os.path.join(img_dir, rand_image))
            axs[i, j].imshow(image)
            axs[i, j].label_outer()
    plt.show()
    return None
