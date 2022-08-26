import os
import os.path as osp
import pickle as rick
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url
import pandas as pd
import subprocess
import numpy as np
import csv
import argparse
import random


class MyDataset(Dataset):
    def __init__(self, root, input_list_size, content_raw, path_to_processed, transform=None, pre_transform=None):
        # root = Where the Dataset should be stored. This Folder is
        # split into raw_dir (where the raw data should be or is downloaded to)
        # and processed_dir (where the processed data is saved).
        # I use a path to the raw data iin the process function and don't
        # use the raw_dir.
        self.input_list_size = input_list_size
        self.content_raw = content_raw
        self.path_to_processed = path_to_processed

        super().__init__(root, transform, pre_transform)
        # It can be that it should be:
        # super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.content_raw

    @property
    def processed_file_names(self):
        content_processed = os.listdir(self.path_to_processed)
        if len(content_processed) > 10:
            file_names = content_processed
            try:
                file_names.remove("pre_filter.pt")
                file_names.remove("pre_transform.pt")
            except:
                print("No prefilter and pretransform files")
        else:
            file_names = ["data_1.pt", "data_2.pt"]
        return file_names

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # print(raw_path)   # debugging
            pickle_in = open(raw_path, "rb")
            for step in range(0, self.input_list_size):
                try:
                    entry = rick.load(pickle_in)
                    features = torch.tensor(entry[0], dtype=torch.float)
                    edge_index = torch.tensor(entry[1], dtype=torch.long)
                    edge_index = edge_index.t().contiguous()
                    attr = torch.tensor(entry[2])
                    edge_attr = attr.float()  # necessary, because edge_attr must be in different
                    # shape, than edge_weight
                    label = entry[3]
                    #label_tensor = torch.tensor([label], dtype=torch.float)  # float for Regression
                    label_tensor = torch.tensor([label], dtype=torch.long)  # Long for Classification
                    intarna_energy = entry[4]
                    intarna_energy_tensor = torch.tensor([intarna_energy])
                    covalent_edges = (edge_attr[:, 0] == 1)
                    #if entry[5] == "train":
                    #    split = torch.tensor([0])
                    #elif entry[5] == "val":
                    #    split = torch.tensor([1])
                    #rna_ids = torch.tensor(entry[6])
                    data_entry = Data(x=features,
                                      edge_index=edge_index,
                                      edge_attr=edge_attr,
                                      y=label_tensor,
                                      intarna_energy=intarna_energy_tensor,
                                      covalent_edges=covalent_edges
                                      )
                    torch.save(data_entry,
                               osp.join(self.processed_dir,
                                        f'data_{idx}.pt'))
                    if idx % 100 == 0:
                        print(idx)
                    idx = idx + 1
                except:
                    break
            pickle_in.close()
        idx = idx - 1
        print(idx)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   f'data_{idx}.pt'))
        return data


def read_file(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=[1])
    return df


def read_genome(file_name):
    with open(file_name) as file:
        genome = file.readlines()    # lines of file a read in a list
        genome_string = "".join(str(line) for line in genome[1:])  # merges the lines into one string (without first line(header))
        genome_string = genome_string.replace("\n", "")   # remove \n from the string
    return genome_string


# Specific for RIL-seq dataset from Melamed et al. 2016.
def extract_sequence_etc(entry, genome, label, headers):
    rna_1 = entry["RNA1 name"]
    rna_1_from = entry["RNA1 from"]
    rna_1_to = entry["RNA1 to"]
    rna_1_strand = entry["RNA1 strand"]

    rna_2 = entry["RNA2 name"]
    rna_2_from = entry["RNA2 from"]
    rna_2_to = entry["RNA2 to"]
    rna_2_strand = entry["RNA2 strand"]

    # Make sure, the sRNA is always at position 2:
    if entry["Genomic annotation of RNA1"] == "sRNA" and entry["Genomic annotation of RNA1"] != "sRNA":
        rna_1, rna_2 = rna_2, rna_1
        rna_1_from, rna_2_from = rna_2_from, rna_1_from
        rna_1_to, rna_2_to = rna_2_to, rna_1_to
        rna_1_strand, rna_2_strand = rna_2_strand, rna_1_strand

    rna_1_from = rna_1_from - 50
    rna_1_to = rna_1_to + 50
    rna_2_from = rna_2_from - 30
    rna_2_to = rna_2_to + 30

    #rna_1 = entry["RNA1 name"]
    #rna_1_from = entry["RNA1 from"] - 50
    #rna_1_to = entry["RNA1 to"] + 50
    #rna_1_strand = entry["RNA1 strand"]
    sequence_rna_1 = genome[rna_1_from:rna_1_to]
    if rna_1_strand == "-":
        with open("temp_sequence.fasta", "w") as f:
            sequence_rna_1 = ">mock_header\n" + sequence_rna_1
            f.write(sequence_rna_1)
        revseq = "revseq temp_sequence.fasta temp_rev_comp_seq.fasta"
        subprocess.run(revseq, shell=True)
        with open("temp_rev_comp_seq.fasta", "r") as fi:
            rev_comp_seq = fi.readlines()
        subprocess.run("rm temp_rev_comp_seq.fasta", shell=True)
        sequence_rna_1 = rev_comp_seq[1].replace("\n", "")
    #rna_2 = entry["RNA2 name"]
    #rna_2_from = entry["RNA2 from"] - 30
    #rna_2_to = entry["RNA2 to"] + 30
    #rna_2_strand = entry["RNA2 strand"]
    sequence_rna_2 = genome[rna_2_from:rna_2_to]
    if rna_2_strand == "-":
        with open("temp_sequence.fasta", "w") as f:
            sequence_rna_2 = ">mock_header\n" + sequence_rna_2
            f.write(sequence_rna_2)
        revseq = "revseq temp_sequence.fasta temp_rev_comp_seq.fasta"
        subprocess.run(revseq, shell=True)
        with open("temp_rev_comp_seq.fasta", "r") as fi:
            rev_comp_seq = fi.readlines()
        subprocess.run("rm temp_rev_comp_seq.fasta", shell=True)
        sequence_rna_2 = rev_comp_seq[1].replace("\n", "")
    fisher_value = entry["Fisher's exact test p-value"]
    new_entry = pd.DataFrame([[rna_1, rna_2, sequence_rna_1, sequence_rna_2, label, fisher_value]],
                             columns=headers)
    return new_entry


def target_cov_bonds(target_sequence):
    edge_index = []
    edge_attributes = []
    feature_vector = []
    for indx, base in enumerate(target_sequence):
        entry = []
        if base == "A":
            entry = [1, 0, 0, 0]
        elif base == "C":
            entry = [0, 1, 0, 0]
        elif base == "G":
            entry = [0, 0, 1, 0]
        elif base == "T":
            entry = [0, 0, 0, 1]
        elif base == "U":
            entry = [0, 0, 0, 1]
        feature_vector.append(entry)
        if indx <= (len(target_sequence) - 2):  # Because of the indexing from 0, indx (and indx + 1)
            # should never be larger than len(sequence) - 1!
            edge_forward = [indx, (indx + 1)]
            edge_backwards = [(indx + 1), indx]
            edge_index.append(edge_forward)
            attribute = [1, 0, 0, 1]   # The first three: one hot encoded edge type [cov, intra, inter], the last: edge weight
            edge_attributes.append(attribute)
            edge_index.append(edge_backwards)
            edge_attributes.append(attribute)  # Attribut muss sowohl für edge_forward,
            # als auch für edge_backwards angefügt werden.
    return edge_index, edge_attributes, feature_vector


# edge_index, edge_attributes und feature_vector für die Suchsequenz (sRNA) und die kovalenten Bindungen.
def query_cov_bonds(query_sequence, target_sequence, edge_index, edge_attributes, feature_vector):
    for indx, base in enumerate(query_sequence):
        entry = []
        if base == "A":
            entry = [1, 0, 0, 0]
        elif base == "C":
            entry = [0, 1, 0, 0]
        elif base == "G":
            entry = [0, 0, 1, 0]
        elif base == "T":
            entry = [0, 0, 0, 1]
        elif base == "U":
            entry = [0, 0, 0, 1]
        feature_vector.append(entry)
        if indx <= (len(query_sequence) - 2):
            index = (len(target_sequence) - 1) + 1 + indx
            edge_forward = [index, (index + 1)]
            edge_backwards = [(index + 1), index]
            edge_index.append(edge_forward)
            attribute = [1, 0, 0, 1]   # The first three: one hot encoded edge type [cov, intra, inter], the last: edge weight
            edge_attributes.append(attribute)
            edge_index.append(edge_backwards)
            edge_attributes.append(attribute)     # Attribut muss sowohl für edge_forward,
            # als auch für edge_backwards angefügt werden.
    return edge_index, edge_attributes, feature_vector


def use_plfold(sequence):
    if len(sequence) <= 70:
        max_win_size = str(len(sequence) - 1)
        plfold_arguments = "echo " + sequence  + " | RNAplfold" + " --winsize=" + max_win_size
    else:
        plfold_arguments = "echo " + sequence + " | RNAplfold"
    subprocess.run(plfold_arguments, shell=True)
    plfold_data = ""
    with open("plfold_dp.ps", "r") as plfold_file:
        for line in plfold_file:
            plfold_data = plfold_data + line
    word = "start of base pair probability data"
    pos = plfold_data.find(word)
    pos_start = pos + (len(word) + 1)
    word = "showpage"
    pos = plfold_data.find(word)
    pos_end = pos
    plfold_data1 = plfold_data[pos_start:pos_end]  # nur noch interessante plfold Ergebnisse
    return plfold_data1


def plfold_target_in(plfold_target, edge_index, edge_attributes):
    plfold_lines = plfold_target.splitlines()
    for line in plfold_lines:
        entry = line.split(" ")
        plfold_index_1 = int(entry[0]) - 1
        plfold_index_2 = int(entry[1]) - 1
        plfold_value = float(entry[2]) * float(entry[2])
        edge_forward = [plfold_index_1, plfold_index_2]
        edge_backwards = [plfold_index_2, plfold_index_1]
        edge_index.append(edge_forward)
        edge_index.append(edge_backwards)
        attr = [0, 1, 0, plfold_value]
        edge_attributes.append(attr)
        edge_attributes.append(attr)
    return edge_index, edge_attributes


def plfold_query_in(sequence_target, plfold_query, edge_index, edge_attributes):
    length = len(sequence_target)
    plfold_lines_query = plfold_query.splitlines()
    for line in plfold_lines_query:
        entry = line.split(" ")
        plfold_index_1 = int(entry[0]) - 1 + length
        plfold_index_2 = int(entry[1]) - 1 + length
        plfold_value = float(entry[2]) * float(entry[2])
        edge_forward = [plfold_index_1, plfold_index_2]
        edge_backwards = [plfold_index_2, plfold_index_1]
        edge_index.append(edge_forward)
        edge_index.append(edge_backwards)
        attr = [0, 1, 0, plfold_value]
        edge_attributes.append(attr)
        edge_attributes.append(attr)
    return edge_index, edge_attributes


# accessibility=False for IntaRNA to not take accessibility into account. This should be done separately by RNAplfold
# and encoded into intramolecular edges.
# Before the 15.08.22, the computation of accessibility was not disabled!
def use_intarna(target_sequence, query_sequence, output_file, accessibility=False):
    if accessibility:
        intarna_arguments = "IntaRNA -t " + target_sequence + " -q " + query_sequence + " --out spotProb:" + \
                            output_file + " --qIdxPos0=0 --tIdxPos0=0"
        print("Using IntaRNA accessibility computation!")
    else:
        intarna_arguments = "IntaRNA -t " + target_sequence + " -q " + query_sequence + " --out spotProb:" + \
                            output_file + " --qIdxPos0=0 --tIdxPos0=0" + " --acc=N"
    p = subprocess.run(intarna_arguments, shell=True, capture_output=True)
    intarna_stdout = str(p.stdout)
    return intarna_stdout


def intarna_in(edge_index, edge_attributes, output_file):
    with open(output_file) as csvdatei:
        csv_spotprobs = csv.reader(csvdatei, delimiter=";")
        result = list(csv_spotprobs)
    intarna_spotprobs = np.array(result)                   # Wäre wahrscheinlich einfacher, mit Pandas Dataframe, aber so funktionierts auch.
    intarna_spotprobs = intarna_spotprobs[1:, 1:]          # Überschriften werden abgeschnitten.
    intarna_spotprobs = intarna_spotprobs.astype('float')
    length_target = intarna_spotprobs.shape[0]
    intarna_index_1 = 0
    for row in intarna_spotprobs:                        # Loopt über die Reihen und in den Reihen über die Spalten und fügt die Werte und die zugehörigen Indices an edge_index und edge_attributes an.
        intarna_index_2 = 0
        for element in row:
            if element != 0:                             # Wenn der von IntaRNA ausgegebene Wert gleich 0 ist, muss auch nichts eingetragen werden.
                edge_forward = [intarna_index_1, (intarna_index_2 + length_target)]
                edge_backwards = [(intarna_index_2 + length_target), intarna_index_1]
                edge_index.append(edge_forward)
                edge_index.append(edge_backwards)
                attr = [0, 0, 1, element]
                edge_attributes.append(attr)
                edge_attributes.append(attr)
            intarna_index_2 = intarna_index_2 + 1
        intarna_index_1 = intarna_index_1 + 1
    return edge_index, edge_attributes


# UTR (= rna_1) and sRNA (=rna_2) were accidentally switched for some of the functions! This is fixed in the new
# version of this function, but for data that was prepared before 16.08.22, this function with the mistake was used!
def build_graphs_on_the_fly_old(rna_1_seq, rna_2_seq, rna_1_name, rna_2_name, exceptions, label, intarna_accessibility=False):
    no_favorable_interaction = False
    entry_list = []
    output_file = "intarna_spotprob.csv"
    intarna_stdout = use_intarna(rna_1_seq, rna_2_seq, output_file, accessibility=intarna_accessibility)
    if "nno favorable interaction" in intarna_stdout:
        print("No favorable interaction")
        entry_list.append([rna_1_name, rna_2_name])
        no_favorable_interaction = True
    else:
        edge_index, edge_attributes, feature_vector = target_cov_bonds(rna_2_seq)  # Is rna_2 correct? I think it shoul be rna_1!
        edge_index, edge_attributes, feature_vector = query_cov_bonds(rna_1_seq, rna_2_seq, edge_index,
                                                                      edge_attributes, feature_vector) # I think, rna_1 and rna_2 need to be switched!
        try:
            find_intarna_energy = intarna_stdout.find("interaction energy = ")
            start_pos_energy = find_intarna_energy + len("interaction energy = ")
            end_pos_energy = intarna_stdout.find(" kcal/mol")
            interaction_energy = float(intarna_stdout[start_pos_energy:end_pos_energy])
            positive_interaction_energy = interaction_energy * (-1)
            plfold_mrna = use_plfold(rna_2_seq)
            plfold_srna = use_plfold(rna_1_seq)
            edge_index, edge_attributes = plfold_target_in(plfold_mrna, edge_index, edge_attributes)
            edge_index, edge_attributes = plfold_query_in(rna_2_seq, plfold_srna, edge_index, edge_attributes)
            edge_index, edge_attributes = intarna_in(edge_index, edge_attributes, output_file)
            entry_list.append(feature_vector)
            entry_list.append(edge_index)
            entry_list.append(edge_attributes)
            entry_list.append(label)
            entry_list.append(positive_interaction_energy)
            entry_list.append([rna_1_name, rna_2_name])
        except:
            print("exception \n" + intarna_stdout)
            entry_list.append([rna_1_name, rna_2_name])
            no_favorable_interaction = True
    return entry_list, exceptions, no_favorable_interaction


# The switch-up of UTR (= rna_1) and sRNA (=rna_2) for some of the functions is fixed!
def build_graphs_on_the_fly(rna_1_seq, rna_2_seq, rna_1_name, rna_2_name, exceptions, label,
                            intarna_accessibility=False, remove_weak_edges=True):
    edge_weight_threshold = 0.05
    no_favorable_interaction = False
    entry_list = []
    output_file = "intarna_spotprob.csv"
    intarna_stdout = use_intarna(rna_1_seq, rna_2_seq, output_file, accessibility=intarna_accessibility)
    if "nno favorable interaction" in intarna_stdout:
        print("No favorable interaction")
        entry_list.append([rna_1_name, rna_2_name])
        no_favorable_interaction = True
    else:
        edge_index, edge_attributes, feature_vector = target_cov_bonds(rna_1_seq)
        edge_index, edge_attributes, feature_vector = query_cov_bonds(rna_2_seq, rna_1_seq, edge_index,
                                                                      edge_attributes, feature_vector)
        try:
            find_intarna_energy = intarna_stdout.find("interaction energy = ")
            start_pos_energy = find_intarna_energy + len("interaction energy = ")
            end_pos_energy = intarna_stdout.find(" kcal/mol")
            interaction_energy = float(intarna_stdout[start_pos_energy:end_pos_energy])
            positive_interaction_energy = interaction_energy * (-1)
            plfold_mrna = use_plfold(rna_1_seq)
            plfold_srna = use_plfold(rna_2_seq)
            edge_index, edge_attributes = plfold_target_in(plfold_mrna, edge_index, edge_attributes)
            edge_index, edge_attributes = plfold_query_in(rna_1_seq, plfold_srna, edge_index, edge_attributes)
            edge_index, edge_attributes = intarna_in(edge_index, edge_attributes, output_file)
            if remove_weak_edges:
                edge_index_keep = []
                edge_attributes_keep = []
                for i, attr in enumerate(edge_attributes):
                    if attr[3] >= edge_weight_threshold:
                        edge_attributes_keep.append(attr)
                        edge_index_keep.append(edge_index[i])
                edge_index = edge_index_keep
                edge_attributes = edge_attributes_keep
            entry_list.append(feature_vector)
            entry_list.append(edge_index)
            entry_list.append(edge_attributes)
            entry_list.append(label)
            entry_list.append(positive_interaction_energy)
            entry_list.append([rna_1_name, rna_2_name])
        except:
            print(f"RNA 1: {rna_1_name} \n"
                  f"RNA 1 sequence: {rna_1_seq} \n"
                  f"RNA 2: {rna_2_name} \n"
                  f"RNA 2 sequence: {rna_2_seq}")
            print("exception \n" + intarna_stdout)
            entry_list.append([rna_1_name, rna_2_name])
            no_favorable_interaction = True
    return entry_list, exceptions, no_favorable_interaction


def prepare_ril_seq_melamed(number_of_samples="all_positives", fraction_positives=1, fraction_negatives=100):
    # genome_file = "data/e_coli_k12_mg1655_genome.fasta"  # This is Version 3 of the genome! In the RIL-seq paper from
    # 2016, genome version 2 was used!!!
    # Version 2:
    genome_file = "data/e_coli_k12_mg1655_genome_NC_000913_2.fasta"
    ril_seq_file = "data/ril_seq_dataset_melamed_et_al_2016.xlsx"

    base_name_pickle_output_lists = "data/test_ril_seq_23_08_22/raw/ril_seq_melamed_gnn_nested_input_list_"
                                    # (rest of file names is added for each list)
    file_name_no_fav_intarna_interaction = "data/test_ril_seq_23_08_22/no_favorable_interaction_ril_seq_melamed_16_08_22.pkl"
    pickle_list_size = 2000

    genome = read_genome(genome_file)

    ril_seq_table_1 = read_file(ril_seq_file, sheet_name="Log_phase")
    ril_seq_table_2 = read_file(ril_seq_file, sheet_name="Stationary_phase")
    ril_seq_table_3 = read_file(ril_seq_file, sheet_name="Iron_limitation")

    # Build table for building graphs:

    headers = ["rna_1", "rna_2", "rna_1 sequence", "rna_2 sequence", "label", "Fisher's exact test p-value"]
    ril_seq_combined_table = pd.DataFrame(columns=headers)

    for i in range(len(ril_seq_table_1)):
        entry = ril_seq_table_1.loc[i]
        label = 1
        try:
            new_entry = extract_sequence_etc(entry, genome, label, headers)
            ril_seq_combined_table = pd.concat([ril_seq_combined_table, new_entry], ignore_index=True)
        except:
            print(f"Sequence could not be extracted. Row {i + 3} in excel sheet 1")

    for i in range(len(ril_seq_table_2)):
        entry = ril_seq_table_2.loc[i]
        label = 1
        rna_1 = entry["RNA1 name"]
        rna_2 = entry["RNA2 name"]
        if not ((ril_seq_combined_table["rna_1"] == rna_1) & (ril_seq_combined_table["rna_2"] == rna_2)).any():
            if not ((ril_seq_combined_table["rna_1"] == rna_2) & (ril_seq_combined_table["rna_2"] == rna_1)).any():
                try:
                    new_entry = extract_sequence_etc(entry, genome, label, headers)
                    ril_seq_combined_table = pd.concat([ril_seq_combined_table, new_entry], ignore_index=True)
                except:
                    print(f"Sequence could not be extracted. Row {i + 3} in excel sheet 2")

    for i in range(len(ril_seq_table_3)):
        entry = ril_seq_table_3.loc[i]
        label = 1
        rna_1 = entry["RNA1 name"]
        rna_2 = entry["RNA2 name"]
        if not ((ril_seq_combined_table["rna_1"] == rna_1) & (ril_seq_combined_table["rna_2"] == rna_2)).any():
            if not ((ril_seq_combined_table["rna_1"] == rna_2) & (ril_seq_combined_table["rna_2"] == rna_1)).any():
                try:
                    new_entry = extract_sequence_etc(entry, genome, label, headers)
                    ril_seq_combined_table = pd.concat([ril_seq_combined_table, new_entry], ignore_index=True)
                except:
                    print(f"Sequence could not be extracted. Row {i + 3} in excel sheet 3")

    # Create dataframe for negative instances:
    df_negativ_instances = pd.DataFrame(columns=headers)

    # For each sRNA (or rna_2), sample 30 rna_1s that are not interacting in this dataset.
    for i in range(len(ril_seq_combined_table)):
        entry_rna_2 = ril_seq_combined_table.loc[i]
        rna_2_seq = entry_rna_2["rna_2 sequence"]
        rna_2_name = entry_rna_2["rna_2"]
        df_neg_rna_1 = ril_seq_combined_table[(ril_seq_combined_table["rna_2"] != entry_rna_2["rna_2"])]
        df_neg_rna_1 = df_neg_rna_1.sample(30)
        df_neg_rna_1.reset_index(drop=True, inplace=True)
        for j in range(len(df_neg_rna_1)):
            entry_rna_1 = df_neg_rna_1.loc[j]
            rna_1_seq = entry_rna_1["rna_1 sequence"]
            rna_1_name = entry_rna_1["rna_1"]
            # Make sure, this combination of rna_1 and rna_2 does not already exist in the dataframe with the positive
            # instances (ril_seq_combined_table)or the dataframe with the negative instances (df_negativ_instances).
            if not ((ril_seq_combined_table["rna_1"] == rna_1_name) & (
                    ril_seq_combined_table["rna_2"] == rna_2_name)).any():
                if not ((ril_seq_combined_table["rna_1"] == rna_2_name) & (
                        ril_seq_combined_table["rna_2"] == rna_1_name)).any():
                    if not ((df_negativ_instances["rna_1"] == rna_1_name) & (
                            df_negativ_instances["rna_2"] == rna_2_name)).any():
                        if not ((df_negativ_instances["rna_1"] == rna_2_name) & (
                                df_negativ_instances["rna_2"] == rna_1_name)).any():
                            fisher_value = "NA"
                            label = 0
                            new_entry = pd.DataFrame(
                                [[rna_1_name, rna_2_name, rna_1_seq, rna_2_seq, label, fisher_value]],
                                columns=headers)
                            df_negativ_instances = pd.concat([df_negativ_instances, new_entry], ignore_index=True)

    # Sample the number of samples asked for with the correct proportions of positive and negative instances:
    if number_of_samples == "all_positives":
        number_positives = len(ril_seq_combined_table)
        number_negatives = number_positives * fraction_negatives
        negatives = df_negativ_instances.sample(number_negatives)
        ril_seq_combined_table = pd.concat([ril_seq_combined_table, negatives], ignore_index=True)
    else:
        number_positives = round((number_of_samples / (fraction_positives + fraction_negatives)) * fraction_positives)
        number_negatives = round((number_of_samples / (fraction_positives + fraction_negatives)) * fraction_negatives)
        positives = ril_seq_combined_table.sample(number_positives)
        negatives = df_negativ_instances.sample(number_negatives)
        ril_seq_combined_table = pd.concat([positives, negatives], ignore_index=True)

    # Shuffle the dataframe before building and saving the graphs so that the pickle lists are not sorted:
    ril_seq_combined_table = ril_seq_combined_table.sample(frac=1).reset_index(drop=True)

    num_lists = int(len(ril_seq_combined_table) / pickle_list_size)
    print(f"Number of samples: {len(ril_seq_combined_table)} \n"
          f"Number of positive samples: {number_positives} \n"
          f"Number of negative instances: {number_negatives} \n"
          f"Number of pickle lists: {num_lists} \n")

    exceptions = 0

    nr_sequences = 0
    nr_overall_sequences = 0
    start_list = 1

    file_name = "place_holder"

    for index, row in ril_seq_combined_table.iterrows():
        rna_1_seq = row["rna_1 sequence"]
        rna_2_seq = row["rna_2 sequence"]
        rna_1_name = row["rna_1"]
        rna_2_name = row["rna_2"]
        label = row["label"]
        entry_list, exceptions, no_favorable_interaction = build_graphs_on_the_fly(rna_1_seq, rna_2_seq, rna_1_name,
                                                                                   rna_2_name, exceptions, label,
                                                                                   intarna_accessibility=False,
                                                                                   remove_weak_edges=True)
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
            if index % 10 == 0:
                print(index)

    print(f"{nr_overall_sequences} total usable graphs. \n{exceptions} exceptions.")
    return ril_seq_combined_table



