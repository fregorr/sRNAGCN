from matplotlib import pyplot as plt
import pickle as rick
import pandas as pd
import numpy as np


def visualize_metrics(file_name, train_metrics=True, grid=True, baseline_metrics=False):
    with open(file_name, "rb") as pickle_in:
        metrics_dict = rick.load(pickle_in)
    lmeans = metrics_dict["lmeans"]
    #param_dict = metrics_dict["param_dict"]

    epochs = []
    epoch = 0
    if baseline_metrics:
        for i in range(len(lmeans) + 1):
            epochs.append(epoch)
            epoch = epoch + 1
    else:
        for i in range(len(lmeans)):
            epochs.append(epoch)
            epoch = epoch + 1

    linewidth1 = 1

    fig1, ax1 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    if baseline_metrics:
        ax1.plot(epochs[1:], lmeans, color="C3")
    else:
        ax1.plot(epochs, lmeans, color="C3")
    ax1.set_ylabel("mean loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylim([0, 0.75])

    fig1.suptitle(file_name, fontsize=6)

    fig2, (ax2, ax3) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

    if train_metrics:
        ax2.plot(epochs, metrics_dict["train_accs"], color="C1", linewidth=linewidth1)
    ax2.plot(epochs, metrics_dict["val_accs"], color="C8", linewidth=linewidth1)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0, 1.05])

    if train_metrics:
        ax3.plot(epochs, metrics_dict["train_precs"], color="C1", linewidth=linewidth1, label="Training")
    ax3.plot(epochs, metrics_dict["val_precs"], color="C8", linewidth=linewidth1, label="Validation")
    ax3.set_ylabel("Precision")
    ax3.set_xlabel("Epochs")
    ax3.set_ylim([0, 0.2625])
    #ax3.set_ylim([0, 1.05])

    if train_metrics:
        # Legende
        leg = ax3.legend(loc="lower right")
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)

    fig2.suptitle(file_name, fontsize=10)

    fig3, (ax4, ax5, ax6) = plt.subplots(figsize=(15, 5), nrows=1, ncols=3)

    if train_metrics:
        ax4.plot(epochs, metrics_dict["train_recs"], color="C1", linewidth=linewidth1)
    ax4.plot(epochs, metrics_dict["val_recs"], color="C8", linewidth=linewidth1)
    ax4.set_ylabel("Recall")
    ax4.set_xlabel("Epochs")
    ax4.set_ylim([0, 1.05])

    if train_metrics:
        ax5.plot(epochs, metrics_dict["train_specs"], color="C1", linewidth=linewidth1)
    ax5.plot(epochs, metrics_dict["val_specs"], color="C8", linewidth=linewidth1)
    ax5.set_ylabel("Specificity")
    ax5.set_xlabel("Epochs")
    ax5.set_ylim([0, 1.05])

    if train_metrics:
        ax6.plot(epochs, metrics_dict["train_f1s"], color="C1", linewidth=linewidth1, label="Training")
    ax6.plot(epochs, metrics_dict["val_f1s"], color="C8", linewidth=linewidth1, label="Validation")
    ax6.set_ylabel("F1")
    ax6.set_xlabel("Epochs")
    ax6.set_ylim([0, 0.2625])
    #ax6.set_ylim([0, 1.05])

    if train_metrics:
        leg2 = ax6.legend(loc="lower right")
        leg2.get_lines()[0].set_linewidth(2)
        leg2.get_lines()[1].set_linewidth(2)

    fig3.suptitle(file_name, fontsize=10)

    if grid:
        ax1.grid(True, linewidth=0.5)
        ax2.grid(True, linewidth=0.5)
        ax3.grid(True, linewidth=0.5)
        ax4.grid(True, linewidth=0.5)
        ax5.grid(True, linewidth=0.5)
        ax6.grid(True, linewidth=0.5)

    if "val_aurpcs" in metrics_dict:
        fig4, ax7 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

        if train_metrics:
            ax7.plot(epochs, metrics_dict["train_aurpcs"], color="C1", linewidth=linewidth1, label="Training")
        ax7.plot(epochs, metrics_dict["val_aurpcs"], color="C8", linewidth=linewidth1, label="Validation")
        ax7.set_ylabel("area under the precision recall curve")
        ax7.set_xlabel("epochs")
        ax7.set_ylim([0, 0.2625])
        #ax7.set_ylim([0, 1.05])

        if train_metrics:
            leg3 = ax7.legend(loc="lower right")
            leg3.get_lines()[0].set_linewidth(2)
            leg3.get_lines()[1].set_linewidth(2)

        fig4.suptitle(file_name, fontsize=6)

        if grid:
            ax7.grid(True, linewidth=0.5)
    else:
        print("AUPRC is not in metrics dict. ")

    #max_train_f1 = max(metrics_dict["train_f1s"])
    #max_val_f1 = max(metrics_dict["val_f1s"])

    return metrics_dict, fig1, ax1, fig2, ax2, ax3, fig3, ax4, ax5, ax6, fig4, ax7


def calculate_mae(x, y):
    sum_of_errors = 0
    for i, j in enumerate(y):
        err = x[i] - j
        sum_of_errors = sum_of_errors + abs(err)
    mae = sum_of_errors / len(y)
    return mae


def visualize_poddar_metrics(file_name, f_good_epochs_dic, plot_epoch):
    with open(file_name, "rb") as p_in:
        lmeans = rick.load(p_in)
        lmins = rick.load(p_in)
        train_predictions = rick.load(p_in)
        train_labels = rick.load(p_in)
        val_predictions = rick.load(p_in)
        val_labels = rick.load(p_in)
        param_dict = rick.load(p_in)
        val_maes = rick.load(p_in)
        train_mae = rick.load(p_in)
        val_mae = rick.load(p_in)

    with open(f_good_epochs_dic, "rb") as p_in:
        good_epochs_dic = rick.load(p_in)

    epochs = []
    epoch = 0
    for i in range(len(lmeans)):
        epochs.append(epoch)
        epoch = epoch + 1

    fig1, ax1 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    fig3, ax3 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    fig4, ax4 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    fig5, ax5 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)

    ax1.plot(epochs, lmeans)
    ax2.plot(epochs, val_maes)

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("MAE")

    ax1.set_ylim([0, 0.73])
    ax2.set_ylim([0, 0.73])

    ax1.grid(True, linewidth=0.5)
    ax2.grid(True, linewidth=0.5)

    key1 = f"train_preds_epoch_{plot_epoch}"
    key2 = f"train_labels_epoch_{plot_epoch}"
    key3 = f"val_preds_epoch_{plot_epoch}"
    key4 = f"val_labels_epoch_{plot_epoch}"
    key5 = f"test_preds_epoch_{plot_epoch}"
    key6 = f"test_labels_epoch_{plot_epoch}"

    ax3.scatter(good_epochs_dic[key2], good_epochs_dic[key1], s=1)
    ax4.scatter(good_epochs_dic[key4], good_epochs_dic[key3], s=1)
    ax5.scatter(good_epochs_dic[key6], good_epochs_dic[key5], s=1)

    train_mae = calculate_mae(good_epochs_dic[key1], good_epochs_dic[key2])
    val_mae = calculate_mae(good_epochs_dic[key3], good_epochs_dic[key4])
    test_mae = calculate_mae(good_epochs_dic[key5], good_epochs_dic[key6])

    ax3.set_xlabel("labels")
    ax3.set_ylabel("predictions")
    ax4.set_xlabel("labels")
    ax4.set_ylabel("predictions")
    ax5.set_xlabel("labels")
    ax5.set_ylabel("predictions")

    ax3.set_ylim([0.9, 5.1])
    ax3.set_xlim([0.9, 5.1])
    ax4.set_ylim([0.9, 5.1])
    ax4.set_xlim([0.9, 5.1])
    ax5.set_ylim([0.9, 5.1])
    ax5.set_xlim([0.9, 5.1])

    return fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, fig5, ax5, train_mae, val_mae, test_mae




# Stimmt hinten und vorne noch nicht!!!:

def visualize_graph(graph):
    node_feature = graph.x
    edge_index = graph.edge_index
    edge_weight = graph.edge_attr
    label = graph.y

    last_index = 0
    for count, index in enumerate(edge_index[0]):
        if count % 2 == 0 and index != last_index:
            length_rna_1 = int(last_index)
            break
        last_index = index

    node_list = list(range(0, len(node_feature)))
    nodes_rna_1 = node_list[:length_rna_1]
    nodes_rna_2 = node_list[length_rna_1:]

    weighted_edge_list = []
    kov_edge_list = []
    prob_edge_list = []
    less_prob_edge_list = []
    not_very_prob_edge_list = []
    intra_edge_list = []
    inter_edge_list = []

    for i, j in enumerate(edge_index[0]):
        if i < (len(node_feature) -2) * 2:
            this_edge_weight = 2
            kov_tuple = (int(j), int(edge_index[1][i]))
            kov_edge_list.append(kov_tuple)
        else:
            this_edge_weight = edge_weight[i][3]  # weight is at index 3 of edge_attr

        # tuple is needed as weight input for nx.spring_layout():
        edge_tuple = (int(j), int(edge_index[1][i]), this_edge_weight)
        weighted_edge_list.append(edge_tuple)

        # list to draw interactions:
        if edge_weight[i][3] >= 0.5:
            prob_tuple = (int(j), int(edge_index[1][i]))
            prob_edge_list.append(prob_tuple)
        elif edge_weight[i][3] >= 0.1:
            less_prob_tuple = (int(j), int(edge_index[1][i]))
            less_prob_edge_list.append(less_prob_tuple)
        elif edge_weight[i][3] <= 0.099:
            not_very_prob_tuple = (int(j), int(edge_index[1][i]))
            not_very_prob_edge_list.append(not_very_prob_tuple)
        if int(j) <= length_rna_1 and int(edge_index[1][i]) <= length_rna_1:
            intra_tuple = (int(j), int(edge_index[1][i]))
            intra_edge_list.append(intra_tuple)
        elif int(j) > length_rna_1 and int(edge_index[1][i]) > length_rna_1:
            intra_tuple = (int(j), int(edge_index[1][i]))
            intra_edge_list.append(intra_tuple)
        elif int(j) <= length_rna_1 and int(edge_index[1][i]) > length_rna_1:
            inter_tuple = (int(j), int(edge_index[1][i]))
            inter_edge_list.append(inter_tuple)
        elif int(edge_index[1][i]) <= length_rna_1 and int(j) > length_rna_1:
            inter_tuple = (int(j), int(edge_index[1][i]))
            inter_edge_list.append(inter_tuple)



graph = False

if graph:
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edge_list)

    pos = nx.spring_layout(G, weight="weight", iterations=200, seed=123, k=0.1)  # iterations=50  # k=1 ?

    # nx.draw(G, pos=pos)

    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_rna_1, node_size=50, node_color="royalblue")
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_rna_2, node_size=50, node_color="r")
    nx.draw_networkx_edges(G, pos=pos, edgelist=kov_edge_list, width=0.5)
    nx.draw_networkx_edges(G, pos=pos, edgelist=prob_edge_list, width=0.5)
    nx.draw_networkx_edges(G, pos=pos, edgelist=less_prob_edge_list, width=0.1)
    nx.draw_networkx_edges(G, pos=pos, edgelist=not_very_prob_edge_list, width=0.02)

    nx.draw_networkx_edges(G, pos=pos, edgelist=intra_edge_list, width=0.1)
    nx.draw_networkx_edges(G, pos=pos, edgelist=inter_edge_list, width=0.1)

# For 3D visualization!:
# Very good playlist from sentdex (youtube) about 3D plotting with matplotlib.
# In the new matplotlib version, nothing needs to be imported or installed additionally.
threeD = False

if threeD:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 5, 7, 4, 8, 9, 7, 9, 2, 1]
    z = [2, 3, 3, 3, 5, 7, 6, 8, 8, 8]
    ax.scatter(x, y, z, c="r", marker="o")
    ax.plot(x, y, z)

    # remove grid:
    ax.grid(False)
    # remove ticks:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])
    # remove everything except what was actively plotted:
    ax.axis("off")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection="3d")
    x1 = [-1, -2, -4, 6, 2, -7, -2, 3, 1, 5]
    y1 = [3, -10, 3, 5, -4, -4, -2, -8, -1, -3]
    z1 = [8, 7, 6, 5, 4, 3, 2, 1, 9, 7]
    ax2.scatter(x, y, z, c="r", marker="o")
    ax2.scatter(x1, y1, z1, c="b", marker="^")

# Animation of plots:
# https://www.youtube.com/watch?v=bNbN9yoEOdU

# 3D plot could possibly be animated using:
# ax.view_init()

#for i in range(360):
#    ax.view_init(20, i)
#    ax.redraw_in_frame()

from matplotlib.animation import FuncAnimation

animate = False
if animate:
    def update(frame_number):
        angle = frame_number % 360
        ax.view_init(20, angle)
        # ax.redraw_in_frame()


    animation = FuncAnimation(fig, update, interval=10)
    plt.show()




