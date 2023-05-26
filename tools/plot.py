import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tools import read, utils


def learning_metrics(populated, actual, predicted, ax1, ax2, training_loss, validation_loss, display_labels):
    conf_matrix = metrics.multilabel_confusion_matrix(actual, predicted, labels=display_labels)
    print(metrics.classification_report(actual, predicted, output_dict=False, target_names=display_labels))
    # todo classification_report _classification line 545 - labels a mix of strings and floats?
    exit()
    cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)
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
    for i in range(y):
        for j in range(x):
            rand_image = np.random.choice(img_list)
            obj = lookup_table.loc[lookup_table["IMG_ID"] == int(rand_image[:-4])]
            for name in obj.columns:
                if obj[name].item() == 1:
                    axs[i, j].set_title(name, fontsize=10)
            image = utils.load_image(os.path.join(img_dir, rand_image))
            axs[i, j].imshow(image)
            axs[i, j].label_outer()
    plt.show()
    return None
