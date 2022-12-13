import os
import sys
import pandas as pd
import pickle as rick
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import BatchNorm, PNAConv, GNNExplainer
from torch.nn import Dropout
from torch_geometric.utils import dropout_adj
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ReLU

import math
import matplotlib.pyplot as plt
import sklearn

import main
import data_prep
import models


def load_benchmark_data():
    # load sRNAs:
    file_name_sRNAs = "benchmark_coprarna/sRNAs.txt"

    with open(file_name_sRNAs) as file:
        srnas_df = pd.read_csv(file, header=None)

    # load 5'UTRs:
    file_name_utrs = "benchmark_coprarna/NC_000913_upfromstartpos_200_down_100.fa"

    with open(file_name_utrs) as file:
        utr_df = pd.read_csv(file, header=None)

    # load file with true benchmark interactions:
    file_name_benchmark = "benchmark_coprarna/benchmark_23_05_20_modified.xlsx"
    benchmark_df = pd.read_excel(file_name_benchmark)
    return srnas_df, utr_df, benchmark_df


def predict_one_interaction(srna_name, utr_itag, model, device, reverse_order=False):
    model.eval()
    dropout_conv_1_2 = 0
    dropout_conv_rest = 0

    srnas_df, utr_df, benchmark_df = load_benchmark_data()

    srna = ">" + srna_name
    srna_index = srnas_df.index[srnas_df[0] == srna][0] + 1  # Plus 1, because the sequence is in the next line.
    srna_seq = srnas_df.loc[srna_index][0]

    utr = ">" + utr_itag
    utr_index = utr_df.index[utr_df[0] == utr][0] + 1  # Plus 1, because the sequence is in the next line.
    utr_seq = utr_df.loc[utr_index][0]

    if reverse_order:
        srna_seq, utr_seq = utr_seq, srna_seq
        print("Order of RNA1 and RNA2 switched.")

    exceptions = 0
    entry_list, exceptions, no_favorable_interaction = data_prep.build_graphs_on_the_fly(srna_seq, utr_seq,
                                                                                   srna_name,
                                                                                   utr_itag, exceptions)
    if no_favorable_interaction == False:
        features = torch.tensor(entry_list[0], dtype=torch.float)
        edge_index = torch.tensor(entry_list[1], dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        attr = torch.tensor(entry_list[2])
        edge_attr = attr.float()
        intarna_mfe = entry_list[3]
        intarna_energy_tensor = torch.tensor(intarna_mfe)

        graph = Data(x=features,
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     intarna_energy=intarna_energy_tensor)
        actual_data_loader = DataLoader([graph], batch_size=1, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch in actual_data_loader:
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch,
                            dropout_conv_1_2, dropout_conv_rest)
                pred_probas = out.softmax(dim=1).detach().cpu().numpy()
                prob_of_interact = pred_probas[:, 1]

    return out, pred_probas, prob_of_interact


def prepare_benchmark_one_srna(srna_name, benchmark_result_df, test_dir_path, intarna_accessibility=False,
                               remove_weak_edges=True, rna_id_attr=True):

    base_name_pickle_output_lists = test_dir_path + "raw/" \
                                    "gnn_nested_input_list_"  # (rest of file names is added for each list)
    file_name_no_fav_intarna_interaction = test_dir_path + "no_favorable_interaction_" + srna_name + ".pkl"

    srnas_df, utr_df, benchmark_df = load_benchmark_data()

    srna = ">" + srna_name
    srna_index = srnas_df.index[srnas_df[0] == srna][0] + 1  # Plus 1, because the sequence is in the next line.
    srna_seq = srnas_df.loc[srna_index][0]

    headers = ["srna_name", "utr_itag", "graph_index", "probability_no_interaction", "probability_interaction",
               "in_benchmark_list"]

    graph_index = 0

    exceptions = 0
    nr_sequences = 0
    nr_overall_sequences = 0
    start_list = 1
    pickle_list_size = 2000

    file_name = "place_holder"

    for i in range(0, len(utr_df), 2):
        utr_itag = utr_df.loc[i][0]
        utr_itag = utr_itag.replace(">", "")
        utr_seq = utr_df.loc[i + 1][0]

        # Check, if this is a true interaction (according to benchmark file):
        if ((benchmark_df["srna_name"] == srna_name) & (benchmark_df["target_ltag"] == utr_itag)).any():
            label = 1
            in_benchmark_list = True
        else:
            label = 0
            in_benchmark_list = False

        entry_list, exceptions, no_favorable_interaction = data_prep.build_graphs_on_the_fly(utr_seq, srna_seq,
                                                                                             utr_itag, srna_name,
                                                                                             exceptions, label,
                                                                                             intarna_accessibility=
                                                                                             intarna_accessibility,
                                                                                             remove_weak_edges=
                                                                                             remove_weak_edges,
                                                                                             rna_id_attr=rna_id_attr)

        if no_favorable_interaction:
            pickle_out = open(file_name_no_fav_intarna_interaction, "a+b")
            rick.dump(entry_list, pickle_out)
            pickle_out.close()
            exceptions += 1
        else:
            nr_sequences = nr_sequences + 1
            if nr_sequences == 1:
                file_name = base_name_pickle_output_lists + str(start_list) + "_.pkl"
                start_list = start_list + pickle_list_size
            if nr_sequences == pickle_list_size:
                nr_sequences = 0
            pickle_out = open(file_name, "a+b")
            rick.dump(entry_list, pickle_out)
            pickle_out.close()
            nr_overall_sequences = nr_overall_sequences + 1
            if i % 100 == 0:
                print(i / 2)

            prob_no_interaction = "NA"
            prob_interaction = "NA"
            new_entry = pd.DataFrame([[srna_name, utr_itag, graph_index, prob_no_interaction, prob_interaction,
                                       in_benchmark_list]], columns=headers)
            benchmark_result_df = pd.concat([benchmark_result_df, new_entry], ignore_index=True)
            graph_index = graph_index + 1

    print(f"{nr_overall_sequences} total usable graphs. \n{exceptions} exceptions.")
    return benchmark_result_df


def prepare_benchmark():

    headers = ["srna_name", "utr_itag", "graph_index", "probability_no_interaction", "probability_interaction",
               "in_benchmark_list"]
    benchmark_result_df = pd.DataFrame(columns=headers)

    benchmark_srna_list = ["ArcZ", "ChiX", "CyaR", "DicF", "DsrA", "FnrS", "GcvB", "GlmZ", "McaS", "MgrR", "MicA",
                           "MicC", "MicF", "MicL", "OmrAB", "OxyS", "RprA", "RseX", "RybB", "RydC", "RyhB", "SdsR",
                           "SgrS", "Spot42"]

    #benchmark_srna_list = ["ArcZ"]

    for srna_name in benchmark_srna_list:

        # check, if test directory for the sRNA exists. If not, create it:
        test_dir_path = "data/" + "benchmark_" + srna_name + "_test/"
        if not os.path.exists(test_dir_path):
            os.makedirs(test_dir_path)
            os.makedirs(test_dir_path + "raw/")
            os.makedirs(test_dir_path + "processed/")
            print(f"created directory for: {srna_name}")
        benchmark_result_df = prepare_benchmark_one_srna(srna_name, benchmark_result_df, test_dir_path,
                                                         intarna_accessibility=False, remove_weak_edges=True,
                                                         rna_id_attr=False)

    with open("benchmark_result_df.pkl", "wb") as pickle_out:
        rick.dump(benchmark_result_df, pickle_out)

    return benchmark_result_df


def run_benchmark(file_name_model_state, model_name=models.PNAnet4L, srna="all", binary_classification=False):
    gpu_nr = "cuda:0"
    device = torch.device(gpu_nr if torch.cuda.is_available() else "cpu")

    num_node_features = 6
    input_list_size = 2000
    dropout_conv_1_2 = 0
    dropout_conv_rest = 0

    model = main.load_trained_model(file_name_model_state, device, num_node_features, model_name)

    with open("benchmark_result_df.pkl", "rb") as pickle_in:
        benchmark_result_df = rick.load(pickle_in)

    benchmark_srna_list = ["ArcZ", "ChiX", "CyaR", "DicF", "DsrA", "FnrS", "GcvB", "GlmZ", "McaS", "MgrR", "MicA",
                           "MicC", "MicF", "MicL", "OmrAB", "OxyS", "RprA", "RseX", "RybB", "RydC", "RyhB", "SdsR",
                           "SgrS", "Spot42"]

    if srna == "all":
        result_df = benchmark_result_df
    else:
        result_df = benchmark_result_df[benchmark_result_df["srna_name"] == srna]
        result_df = result_df.reset_index(drop=True)

    srna_name = False
    for i in range(len(result_df)):
        current_srna_name = result_df.loc[i]["srna_name"]
        if not current_srna_name == srna_name:
            srna_name = current_srna_name
            rooot = "/data/" + "benchmark_" + current_srna_name + "_test/"

            path_to_here = os.getcwd()
            path_to_raw = path_to_here + rooot + "raw/"
            path_to_processed = path_to_here + rooot + "processed/"

            content_raw = os.listdir(path_to_raw)
            dataset = data_prep.MyBenchmarkDataset(root="." + rooot, input_list_size=input_list_size,
                                                   content_raw=content_raw, path_to_processed=path_to_processed)
            for index, graph in enumerate(dataset):
                b_num = int(graph.b_number)
                b_num = "b" + str(b_num).zfill(4)
                result_df.loc[result_df[result_df["utr_itag"] == b_num].index, "graph_index"] = index


        graph_index = result_df.loc[i]["graph_index"]

        try:
            actual_data_loader = DataLoader([dataset[graph_index]], batch_size=1, shuffle=False, drop_last=False)
            with torch.no_grad():
                for batch in actual_data_loader:
                    model.eval()
                    batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.intarna_energy, batch.batch,
                                batch.covalent_edges, dropout_conv_1_2, dropout_conv_rest)
                    if binary_classification:
                        prob_of_interact = out.detach().cpu()
                        prob_of_interact = float(prob_of_interact[0][0])
                    else:
                        pred_probas = out.softmax(dim=1).detach().cpu().numpy()
                        prob_of_interact = pred_probas[:, 1][0]
                        prob_no_interact = pred_probas[:, 0][0]

                        result_df.loc[i]["probability_no_interaction"] = prob_no_interact

                    result_df.loc[i]["probability_interaction"] = prob_of_interact

                    #prob_of_interact = out[:, 1].detach().cpu().numpy()
                    #prob_no_interact = out[:, 0].detach().cpu().numpy()
        except:
            print(f"Error: Problem with this graph:\n"
                  f"graph_index: {graph_index}, current_srna_name: {current_srna_name}, i: {i}")

        if i % 1000 == 0:
            print(srna_name, i)

    return result_df, dataset


# To evaluate:
evaluate = False
if evaluate:
    srna = "ArcZ"
    df = result_df[result_df["srna_name"] == srna]
    df = df.sort_values(["probability_interaction"], ascending=False)
    df = df.reset_index(drop=True)
    benchmark_interactions = df[df["in_benchmark_list"] == True]
    benchmark_interactions

if evaluate:
    counter = 0
    for srna in benchmark_rank_dic.values():
        for i in srna:
            try:
                if i < 200:
                    counter = counter + 1
            except:
                print(i)


def evaluate_benchmark(result_df, srna="all"):
    benchmark_dic = {}
    benchmark_rank_dic = {}
    if srna == "all":
        benchmark_srna_list = ["ArcZ", "ChiX", "CyaR", "DicF", "DsrA", "FnrS", "GcvB", "GlmZ", "McaS", "MgrR", "MicA",
                               "MicC", "MicF", "MicL", "OmrAB", "OxyS", "RprA", "RseX", "RybB", "RydC", "RyhB", "SdsR",
                               "SgrS", "Spot42"]
    else:
        benchmark_srna_list = [srna]

    for srna in benchmark_srna_list:
        df = result_df[result_df["srna_name"] == srna]
        df = df.sort_values(["probability_interaction"], ascending=False)
        df = df.reset_index(drop=True)
        benchmark_interactions = df[df["in_benchmark_list"] == True]
        benchmark_rank_dic[srna] = benchmark_interactions.index.tolist()
        benchmark_dic[srna] = benchmark_interactions
    return benchmark_dic, benchmark_rank_dic


def plot_metrics_at_rank(result_df, benchmark_rank_dic, num_ranks=100):
    srnas_df, utr_df, benchmark_df = load_benchmark_data()
    num_srnas = len(benchmark_rank_dic.keys())

    ranks = [0] # At rank 0, everything should be 0. The predictions start at rank 1 (which has index 0.....)
    cumulated_true_positives = [0]
    # cumulated positives:
    for rank in range(1, (num_ranks + 1)):
        counter = 0
        for srna in benchmark_rank_dic.values():
            for i in srna:
                if i < rank:
                    counter = counter + 1
        ranks = ranks + [rank]
        cumulated_true_positives = cumulated_true_positives + [counter]

    fig1, ax1 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax1.plot(ranks, cumulated_true_positives, color="C3")
    ax1.set_ylabel("cumulated true positives")
    ax1.set_xlabel("rank")
    ax1.set_ylim([0, 21])

    # recall, precision and MCC:
    sum_true_positives = len(benchmark_df)
    recalls = [0.0]
    precisions = [0.0]
    mccs = [0.0]
    for rank in range(1, (num_ranks + 1)):
        tp = cumulated_true_positives[rank - 1]
        fp = (rank * num_srnas) - tp
        fn = sum_true_positives - tp
        tn = len(result_df) - rank - fn
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        mcc = (tp * tn) - (fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        recalls = recalls + [recall]
        precisions = precisions + [precision]
        mccs = mccs + [mcc]

    fig2, ax2 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax2.plot(ranks, recalls, color="C3")
    ax2.set_ylabel("recall")
    ax2.set_xlabel("rank")
    ax2.set_ylim([0, 0.105])

    fig3, ax3 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax3.plot(ranks, precisions, color="C3")
    ax3.set_ylabel("precision")
    ax3.set_xlabel("rank")
    ax3.set_ylim([0, 0.021])

    fig4, ax4 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax4.plot(ranks, mccs, color="C3")
    ax4.set_ylabel("MCC")
    ax4.set_xlabel("rank")

    return ranks, fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, cumulated_true_positives, recalls, precisions



def plot_prc(result_df):
    srnas_df, utr_df, benchmark_df = load_benchmark_data()
    baseline_auprc = len(benchmark_df) / len(result_df)

    # y = np array with labels (0 or 1)
    y = result_df["in_benchmark_list"].tolist()
    for i in range(len(y)):
        if y[i] == False:
            y[i] = 0
        elif y[i] == True:
            y[i] = 1
    # probas_pred
    probas_pred = result_df["probability_interaction"].tolist()

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, probas_pred)
    baseline = [baseline_auprc] * len(recall)

    # plot
    fig5, ax5 = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax5.plot(recall, precision, color="C3")
    ax5.plot(recall, baseline, color="lightgrey")
    ax5.set_ylabel("precision")
    ax5.set_xlabel("recall")
    ax5.set_ylim([0, 0.01])

    auprc = sklearn.metrics.average_precision_score(y, probas_pred)

    return auprc, baseline_auprc



