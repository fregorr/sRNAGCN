from matplotlib import pyplot as plt
import pickle as rick
import pandas as pd
import numpy as np


def visualize_metrics(file_name):
    with open(file_name, "rb") as pickle_in:
        metrics_dict = rick.load(pickle_in)
    lmeans = metrics_dict["lmeans"]
    param_dict = metrics_dict["param_dict"]

    epochs = []
    epoch = 0
    for i in range(len(lmeans)):
        epochs.append(epoch)
        epoch = epoch + 1

    linewidth1 = 1

    fig1, ax1 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

    ax1.plot(epochs, lmeans, color="C3")
    ax1.set_ylabel("mean loss")
    ax1.set_xlabel("epochs")

    fig1.suptitle(file_name, fontsize=6)

    fig2, (ax2, ax3) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

    ax2.plot(epochs, metrics_dict["train_accs"], color="C1", linewidth=linewidth1)
    ax2.plot(epochs, metrics_dict["val_accs"], color="C8", linewidth=linewidth1)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0, 1.0])

    ax3.plot(epochs, metrics_dict["train_precs"], color="C1", linewidth=linewidth1, label="Training")
    ax3.plot(epochs, metrics_dict["val_precs"], color="C8", linewidth=linewidth1, label="Validation")
    ax3.set_ylabel("Precision")
    ax3.set_xlabel("Epochs")
    ax3.set_ylim([0, 1.0])

    leg = ax3.legend(loc="lower right")
    leg.get_lines()[0].set_linewidth(2)
    leg.get_lines()[1].set_linewidth(2)

    fig2.suptitle(file_name, fontsize=10)

    fig3, (ax4, ax5, ax6) = plt.subplots(figsize=(15, 5), nrows=1, ncols=3)

    ax4.plot(epochs, metrics_dict["train_recs"], color="C1", linewidth=linewidth1)
    ax4.plot(epochs, metrics_dict["val_recs"], color="C8", linewidth=linewidth1)
    ax4.set_ylabel("Recall")
    ax4.set_xlabel("Epochs")
    ax4.set_ylim([0, 1.0])

    ax5.plot(epochs, metrics_dict["train_specs"], color="C1", linewidth=linewidth1)
    ax5.plot(epochs, metrics_dict["val_specs"], color="C8", linewidth=linewidth1)
    ax5.set_ylabel("Specificity")
    ax5.set_xlabel("Epochs")
    ax5.set_ylim([0, 1.0])

    ax6.plot(epochs, metrics_dict["train_f1s"], color="C1", linewidth=linewidth1, label="Training")
    ax6.plot(epochs, metrics_dict["val_f1s"], color="C8", linewidth=linewidth1, label="Validation")
    ax6.set_ylabel("F1")
    ax6.set_xlabel("Epochs")
    ax6.set_ylim([0, 1.0])

    leg2 = ax6.legend(loc="lower right")
    leg2.get_lines()[0].set_linewidth(2)
    leg2.get_lines()[1].set_linewidth(2)

    # Legende

    fig3.suptitle(file_name, fontsize=10)

    max_train_f1 = max(metrics_dict["train_f1s"])
    max_val_f1 = max(metrics_dict["val_f1s"])

    return param_dict, max_train_f1, max_val_f1