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
from tqdm import tqdm


# To do: implement indexing of interactions
class MyDataset(Dataset):
    def __init__(self, root, input_list_size, content_raw, path_to_processed, interaction_index=False,
                 transform=None, pre_transform=None):
        # root = Where the Dataset should be stored. This Folder is
        # split into raw_dir (where the raw data should be or is downloaded to)
        # and processed_dir (where the processed data is saved).
        # I use a path to the raw data iin the process function and don't
        # use the raw_dir.
        self.input_list_size = input_list_size
        self.content_raw = content_raw
        self.path_to_processed = path_to_processed
        self.interaction_index = interaction_index

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
                    if self.interaction_index:
                        interaction_idx = torch.tensor(entry[6])
                        data_entry = Data(x=features,
                                          edge_index=edge_index,
                                          edge_attr=edge_attr,
                                          y=label_tensor,
                                          intarna_energy=intarna_energy_tensor,
                                          covalent_edges=covalent_edges,
                                          interaction_index=interaction_idx
                                          )
                    else:
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


class MyBenchmarkDataset(Dataset):
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
                    b_number = entry[5][0]
                    b_number = int(b_number[1:])
                    b_number = torch.tensor([b_number])
                    data_entry = Data(x=features,
                                      edge_index=edge_index,
                                      edge_attr=edge_attr,
                                      y=label_tensor,
                                      intarna_energy=intarna_energy_tensor,
                                      covalent_edges=covalent_edges,
                                      b_number=b_number
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


# Has a bug, only first line is returned for multi-line sequences. function now replaced by
# revcom!
def reverse_seq(sequence):
    with open("temp_sequence.fasta", "w") as f:
        sequence = ">mock_header\n" + sequence
        f.write(sequence)
    revseq = "revseq temp_sequence.fasta temp_rev_comp_seq.fasta"
    subprocess.run(revseq, shell=True)
    with open("temp_rev_comp_seq.fasta", "r") as fi:
        rev_comp_seq = fi.readlines()
    subprocess.run("rm temp_rev_comp_seq.fasta temp_sequence.fasta", shell=True)
    reversed_sequence = rev_comp_seq[1].replace("\n", "")
    return reversed_sequence


def complement(sequence):
    basecomplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    letters = list(sequence)
    letters = [basecomplement[base] for base in letters]
    return ''.join(letters)


def revcom(sequence):
    return complement(sequence[::-1])


def load_benchmark_data():
    # load sRNAs:
    file_name_sRNAs = "benchmark_coprarna/sRNAs.txt"

    with open(file_name_sRNAs) as file:
        srnas_table = pd.read_csv(file, header=None)

    names = []
    sequences = []
    for rna in range(0, len(srnas_table), 2):
        name = srnas_table.loc[rna][0][1:]
        seq = srnas_table.loc[rna + 1][0]
        names = names + [name]
        sequences = sequences + [seq]

    ecocyc_ids = ["G0-8871", "G0-9382", "G0-8878", "EG31115", "G7047", "G0-10677", "G0-8867", "G0-8873",
                  "G0-8899", "G0-10671", "G0-8866", "G0-8901", "EG30063", "G0-16601", "G0-8868", "EG31116",
                  "G0-8863", "G0-10574", "G0-8880", "G0-10592", "G0-8872", "G0-8883", "G0-9941", "EG30098"]

    locus_tags = ["b4450", "b4585", "b4438", "b1574", "b1954", "b4699", "b4443", "b4456",
                  "b4426", "b4698", "b4442", "b4427", "b4439", "b4717", "b4444", "b4458",
                  "b4431", "b4603", "b4417", "b4597", "b4451", "b4433", "b4577", "b3864"]

    srnas_df = pd.DataFrame({"name": names, "ecocyc_id": ecocyc_ids,
                             "locus_tag": locus_tags, "sequence": sequences})

    # load 5'UTRs:
    file_name_utrs = "benchmark_coprarna/NC_000913_upfromstartpos_200_down_100.fa"

    with open(file_name_utrs) as file:
        utr_table = pd.read_csv(file, header=None)

    names = []
    sequences = []
    for rna in range(0, len(utr_table), 2):
        name = utr_table.loc[rna][0][1:]
        seq = utr_table.loc[rna + 1][0]
        names = names + [name]
        sequences = sequences + [seq]

    utr_df = pd.DataFrame({"locus_tag": names, "sequence": sequences})

    # load file with true benchmark interactions:
    file_name_benchmark = "benchmark_coprarna/benchmark_23_05_20_modified.xlsx"
    benchmark_df = pd.read_excel(file_name_benchmark)
    srna_ltags = []
    for i in range(len(benchmark_df)):
        srna = benchmark_df["srna_name"].loc[i]
        srna_ltag = srnas_df[srnas_df["name"] == srna]["locus_tag"].iloc[0]
        srna_ltags = srna_ltags + [srna_ltag]
    benchmark_df.insert(loc = 1, column="srna_ltag", value=srna_ltags)
    return srnas_df, utr_df, benchmark_df


def get_id_conversion_files():
    file_name = "data/ecoli_gene_names_b_num_ecocyc_id"
    with open(file_name, "r") as f:
        lines = f.readlines()

    headers = ["ecocyc_id", "b_number", "eck_number"]
    id_look_up_df = pd.DataFrame(columns=headers)

    for i, line in enumerate(lines):
        id_list = line.split()
        try:
            ecocyc_id = id_list[-1]
            b_number = id_list[-3][:-1]
            eck_number = id_list[-2]
            if b_number[0] == "b" and eck_number[:3] == "ECK":
                id_entry = pd.DataFrame([[ecocyc_id, b_number, eck_number]],
                                        columns=headers)
                id_look_up_df = pd.concat([id_look_up_df, id_entry], ignore_index=True)
            else:
                pass
        except:
            pass
            #print(f"Line at index: {i} is not containing the correct ids.\n"
            #      f"The line: {line}\n")

    e_coli_genes_file = "data/e_coli_all_genes_NC_000913_3.fasta"
    headers = ["ids", "sequence"]
    e_coli_all_genes_table = pd.DataFrame(columns=headers)
    header = False
    with open(e_coli_genes_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == ">":
                if header:
                    sequence_entry = pd.DataFrame([[header, sequence]],
                                                  columns=headers)
                    e_coli_all_genes_table = pd.concat([e_coli_all_genes_table, sequence_entry], ignore_index=True)
                header = line[:-1]
                current = "header"
            elif current == "header":
                sequence = line[:-1]
                current = "sequence"
            elif current == "sequence":
                continue_seq = line[:-1]
                sequence = sequence + continue_seq

    file_name_utrs = "benchmark_coprarna/NC_000913_upfromstartpos_200_down_100.fa"

    #with open(file_name_utrs) as file:
        #b_num_up_down_df = pd.read_csv(file, header=None)

    headers = ["b_number", "sequence"]
    b_num_up_down_df = pd.DataFrame(columns=headers)
    header = False
    with open(file_name_utrs, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == ">":
                if header:
                    sequence_entry = pd.DataFrame([[header, sequence]],
                                                  columns=headers)
                    b_num_up_down_df = pd.concat([b_num_up_down_df, sequence_entry], ignore_index=True)
                header = line[1:-1]
                current = "header"
            elif current == "header":
                sequence = line[:-1]
                current = "sequence"
            elif current == "sequence":
                continue_seq = line[:-1]
                sequence = sequence + continue_seq

    return id_look_up_df, e_coli_all_genes_table, b_num_up_down_df


def retrieve_seq_by_b_num(entry, genome, label, headers, id_look_up_df, e_coli_all_genes_table, b_num_up_down_df):
    rna_1 = entry["RNA1 name"]
    rna_1_from = entry["RNA1 from"]
    rna_1_to = entry["RNA1 to"]
    rna_1_strand = entry["RNA1 strand"]
    rna_1_ecocyc_id = entry["RNA1 EcoCyc ID"]

    rna_2 = entry["RNA2 name"]
    rna_2_from = entry["RNA2 from"]
    rna_2_to = entry["RNA2 to"]
    rna_2_strand = entry["RNA2 strand"]
    rna_2_ecocyc_id = entry["RNA2 EcoCyc ID"]

    # Make sure, the sRNA is always at position 2:
    if entry["Genomic annotation of RNA1"] == "sRNA" and entry["Genomic annotation of RNA2"] != "sRNA":
        rna_1, rna_2 = rna_2, rna_1
        rna_1_from, rna_2_from = rna_2_from, rna_1_from
        rna_1_to, rna_2_to = rna_2_to, rna_1_to
        rna_1_strand, rna_2_strand = rna_2_strand, rna_1_strand
        rna_1_ecocyc_id, rna_2_ecocyc_id = rna_2_ecocyc_id, rna_1_ecocyc_id

    #print(f"rna_1_from: {rna_1_from}, \n"
    #      f"rna_1_to: {rna_1_to}, \n"
    #      f"rna_2_from: {rna_2_from}, \n"
    #      f"rna_2_to: {rna_2_to}\n")  # debugging

    if not rna_1_from <= 50:
        rna_1_from = rna_1_from - 50
    if not rna_1_to >= (len(genome) - 50):
        rna_1_to = rna_1_to + 50
    if not rna_2_from <= 50:
        rna_2_from = rna_2_from - 30
    if not rna_2_to >= (len(genome) - 30):
        rna_2_to = rna_2_to + 30

    #print(f"rna_1_from: {rna_1_from}, \n"
    #      f"rna_1_to: {rna_1_to}, \n"
    #      f"rna_2_from: {rna_2_from}, \n"
    #      f"rna_2_to: {rna_2_to}\n")  # debugging

    # Refine usage of additional EcoCyc ID usage! UTR, IGR, IGT...
    # Especially for special cases like 3UTR, IGT, IGR or number at the end.
    rna_1_ecocyc_id_list = rna_1_ecocyc_id.split(sep=".")
    b_num_rna_1 = "not defined"
    if rna_1_ecocyc_id_list[-1] == "AS":

        # reverse and complement, from all_genes_file?

        rna_1_ecocyc_id = rna_1_ecocyc_id_list[0]
        sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_1_ecocyc_id)]
        if sequence_entry.empty == False:
            # reverse the sequence because of Antisense:
            sequence_rna_1 = sequence_entry["sequence"].iloc[0]
            sequence_rna_1 = revcom(sequence_rna_1)
        else:
            sequence_rna_1 = genome[rna_1_from:rna_1_to]
            if rna_1_strand == "-":
                # reverse seq because "-" strand:
                sequence_rna_1 = revcom(sequence_rna_1)

    elif rna_1_ecocyc_id_list[-1] == "IGR" or rna_1_ecocyc_id_list[-1] == "IGT":
        # get sequences based on indices?

        sequence_rna_1 = genome[rna_1_from:rna_1_to]
        #print(f"len sequence: {len(sequence_rna_1)}")  # debugging
        if rna_1_strand == "-":
            sequence_rna_1 = revcom(sequence_rna_1)
        #print(f"len sequence: {len(sequence_rna_1)}")  # debugging

    elif rna_1_ecocyc_id.__contains__("3UTR"):
        sequence_rna_1 = genome[rna_1_from:rna_1_to]
        #print(f"len sequence: {len(sequence_rna_1)}")  # debugging
        if rna_1_strand == "-":
            sequence_rna_1 = revcom(sequence_rna_1)
        #print(f"len sequence: {len(sequence_rna_1)}")  # debugging

        # try to find locus tag!
        ecocyc_id_3utr = rna_1_ecocyc_id_list[0]
        b_num_rna_1 = id_look_up_df.loc[id_look_up_df["ecocyc_id"] == ecocyc_id_3utr]["b_number"].iloc[0]

    else:
        # Get EcoCyc ID, convert to b-number, get sequence from b-number file for genes:

        rna_1_ecocyc_id = rna_1_ecocyc_id_list[0]
        id_entry = id_look_up_df.loc[id_look_up_df["ecocyc_id"] == rna_1_ecocyc_id]
        if id_entry.empty == False:
            rna_1_b_num = id_entry["b_number"].iloc[0]
            b_num_rna_1 = rna_1_b_num
            sequence_entry = b_num_up_down_df.loc[b_num_up_down_df["b_number"] == rna_1_b_num]
            if sequence_entry.empty == False:  # Not all b-numbers are in the b_num_up_down_df-file (I think, mainly sRNAs are missing)
                sequence_rna_1 = sequence_entry["sequence"].iloc[0]
            else:
                sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_1_ecocyc_id)]
                if sequence_entry.empty == False:  # Some EcoCyc IDs are not in the e_coli_all_genes_table
                    sequence_rna_1 = sequence_entry["sequence"].iloc[0]
                else:
                    # Last possibility: get sequences based on indices:
                    sequence_rna_1 = genome[rna_1_from:rna_1_to]
                    if rna_1_strand == "-":
                        sequence_rna_1 = revcom(sequence_rna_1)

        else:
            sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_1_ecocyc_id)]
            if sequence_entry.empty == False:  # Some EcoCyc IDs are not in the e_coli_all_genes_table
                sequence_rna_1 = sequence_entry["sequence"].iloc[0]
            else:
                # Last possibility: get sequences based on indices:
                sequence_rna_1 = genome[rna_1_from:rna_1_to]
                if rna_1_strand == "-":
                    sequence_rna_1 = revcom(sequence_rna_1)

    rna_2_ecocyc_id_list = rna_2_ecocyc_id.split(sep=".")
    b_num_rna_2 = "not defined"
    if rna_2_ecocyc_id_list[-1] == "AS":

        # reverse and complement, from all_genes_file?

        rna_2_ecocyc_id = rna_2_ecocyc_id_list[0]
        sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_2_ecocyc_id)]
        if sequence_entry.empty == False:
            # reverse seq because of AS:
            sequence_rna_2 = sequence_entry["sequence"].iloc[0]
            sequence_rna_2 = revcom(sequence_rna_2)

        else:
            sequence_rna_2 = genome[rna_2_from:rna_2_to]
            if rna_2_strand == "-":
                sequence_rna_2 = revcom(sequence_rna_2)

    elif rna_2_ecocyc_id_list[-1] == "IGR" or rna_2_ecocyc_id_list[-1] == "IGT":
        # get sequences based on indices?

        sequence_rna_2 = genome[rna_2_from:rna_2_to]
        if rna_2_strand == "-":
            sequence_rna_2 = revcom(sequence_rna_2)

    elif rna_2_ecocyc_id.__contains__("3UTR"):
        sequence_rna_2 = genome[rna_2_from:rna_2_to]
        #print(f"len sequence: {len(sequence_rna_2)}")  # debugging
        if rna_2_strand == "-":
            sequence_rna_2 = revcom(sequence_rna_2)
        #print(f"len sequence: {len(sequence_rna_1)}")  # debugging
        # try to find locus tag!
        ecocyc_id_3utr_2 = rna_2_ecocyc_id_list[0]
        b_num_rna_2 = id_look_up_df.loc[id_look_up_df["ecocyc_id"] == ecocyc_id_3utr_2]["b_number"].iloc[0]

    else:
        # Get EcoCyc ID, convert to b-number, get sequence from b-number file for genes:
        rna_2_ecocyc_id = rna_2_ecocyc_id_list[0]
        id_entry = id_look_up_df.loc[id_look_up_df["ecocyc_id"] == rna_2_ecocyc_id]
        if id_entry.empty == False:
            rna_2_b_num = id_entry["b_number"].iloc[0]
            b_num_rna_2 = rna_2_b_num
            sequence_entry = b_num_up_down_df.loc[b_num_up_down_df["b_number"] == rna_2_b_num]
            if sequence_entry.empty == False:  # Not all b-numbers are in the b_num_up_down_df-file (I think, mainly sRNAs are missing)
                sequence_rna_2 = sequence_entry["sequence"].iloc[0]
            else:
                sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_2_ecocyc_id)]
                if sequence_entry.empty == False:  # Some EcoCyc IDs are not in the e_coli_all_genes_table
                    sequence_rna_2 = sequence_entry["sequence"].iloc[0]
                else:
                    # Last possibility: get sequences based on indices:
                    sequence_rna_2 = genome[rna_2_from:rna_2_to]
                    if rna_2_strand == "-":
                        sequence_rna_2 = revcom(sequence_rna_2)

        else:
            sequence_entry = e_coli_all_genes_table.loc[e_coli_all_genes_table["ids"].str.contains(rna_2_ecocyc_id)]
            if sequence_entry.empty == False:  # Some EcoCyc IDs are not in the e_coli_all_genes_table
                sequence_rna_2 = sequence_entry["sequence"].iloc[0]
            else:
                # Last possibility: get sequences based on indices:
                sequence_rna_2 = genome[rna_2_from:rna_2_to]
                if rna_2_strand == "-":
                    sequence_rna_2 = revcom(sequence_rna_2)

    fisher_value = entry["Fisher's exact test p-value"]
    new_entry = pd.DataFrame([[rna_1, rna_2, sequence_rna_1, sequence_rna_2, label, fisher_value, b_num_rna_1,
                               b_num_rna_2, rna_1_ecocyc_id, rna_2_ecocyc_id]],
                             columns=headers)
    return new_entry


def load_ril_seq_melamed(genome, id_look_up_df, e_coli_all_genes_table, b_num_up_down_df):
    extract_seq_by_index = False

    ril_seq_file = "data/ril_seq_dataset_melamed_et_al_2016.xlsx"
    ril_seq_table_1 = read_file(ril_seq_file, sheet_name="Log_phase")
    ril_seq_table_2 = read_file(ril_seq_file, sheet_name="Stationary_phase")
    ril_seq_table_3 = read_file(ril_seq_file, sheet_name="Iron_limitation")

    ril_seq_all = pd.concat([ril_seq_table_1, ril_seq_table_2, ril_seq_table_3],
                            ignore_index=True)
    ril_seq_all = ril_seq_all.drop_duplicates(subset="Code", keep="first",
                                              ignore_index=True)

    # Build table for building graphs:

    headers = ["rna_1", "rna_2", "rna_1 sequence", "rna_2 sequence", "label", "Fisher's exact test p-value",
               "rna_1_b_num", "rna_2_b_num", "rna_1_ecocyc_id", "rna_2_ecocyc_id"]
    ril_seq_combined_table = pd.DataFrame(columns=headers)

    for i in range(len(ril_seq_all)):
        entry = ril_seq_all.loc[i]
        label = 1
        #try:
        if extract_seq_by_index:
            new_entry = extract_sequence_etc(entry, genome, label, headers)
        else:
            new_entry = retrieve_seq_by_b_num(entry, genome, label, headers,
                                              id_look_up_df, e_coli_all_genes_table, b_num_up_down_df)
        ril_seq_combined_table = pd.concat([ril_seq_combined_table, new_entry], ignore_index=True)
        #except:
        #    print(f"Sequence could not be extracted. Row {i + 3} in excel sheet 1")

    return ril_seq_combined_table


def intarna_for_energies(target_sequences_fasta, query_sequence, accessibility, threads):
    output_file = "multi_intarna_all_energies.csv"  # one output-file will be created with IDs and energies
    intarna_arguments = "IntaRNA -t " + target_sequences_fasta + " -q " + query_sequence + " --outMode=C" + \
                        " --outCsvCols=id1,id2,E" + " --out " + output_file + " --acc=" + accessibility + \
                        " --threads=" + str(threads)
    subprocess.run(intarna_arguments, shell=True)
    intarna_df = pd.read_csv(output_file, sep=";")
    return intarna_df


def intarna_ranking(utr_df, rna_2_seq, accessibility, rna_2_counts, fold_sampling_top_10,
                    fold_sampling_top_10_to_50, fold_sampling_rest, threads):
    # write sequences in fasta for multi-thread IntaRNA:
    with open("temp_intarna.fasta", "w") as f:
        fasta = ""
        for utr in utr_df.iterrows():
            entry = ">" + utr[1].iloc[0] + "\n" + utr[1].iloc[1] + "\n"
            fasta = fasta + entry
        f.write(fasta)

    intarna_df = intarna_for_energies("temp_intarna.fasta", rna_2_seq, accessibility, threads)

    # Split the UTRs in bins according to IntaRNA energies (Top 10, Top 50, Rest).
    intarna_df = intarna_df.sort_values("E")
    top_10_df = intarna_df[:round(len(intarna_df) / 10)].reset_index(drop=True)
    top_10_to_50_df = intarna_df[round(len(intarna_df) / 10):round(len(intarna_df) / 2)].reset_index(drop=True)
    rest_df = intarna_df[round(len(intarna_df) / 2):].reset_index(drop=True)

    # Sample (negative) interactions from the bins and collect them in a df:

    n_top_10_samples = rna_2_counts * fold_sampling_top_10
    n_top_10_to_50_samples = rna_2_counts * fold_sampling_top_10_to_50
    n_rest_samples = rna_2_counts * fold_sampling_rest

    if n_top_10_samples <= len(top_10_df):
        df1 = top_10_df.sample(n_top_10_samples, random_state=123)
    else:
        df1 = top_10_df
    if n_top_10_to_50_samples <= len(top_10_to_50_df):
        df2 = top_10_to_50_df.sample(n_top_10_to_50_samples, random_state=123)
    else:
        df2 = top_10_to_50_df
    if n_rest_samples <= len(rest_df):
        df3 = rest_df.sample(n_rest_samples, random_state=123)
    else:
        df3 = rest_df
    df = pd.concat([df1, df2, df3])

    return df


def target_cov_bonds(target_sequence, rna_id_attribute=False):
    edge_index = []
    edge_attributes = []
    feature_vector = []
    for indx, base in enumerate(target_sequence):
        entry = []
        if rna_id_attribute:
            if base == "A":
                entry = [1, 0, 0, 0, 1, 0]
            elif base == "C":
                entry = [0, 1, 0, 0, 1, 0]
            elif base == "G":
                entry = [0, 0, 1, 0, 1, 0]
            elif base == "T":
                entry = [0, 0, 0, 1, 1, 0]
            elif base == "U":
                entry = [0, 0, 0, 1, 1, 0]
        else:
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
        # construct edges:
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
def query_cov_bonds(query_sequence, target_sequence, edge_index, edge_attributes, feature_vector,
                    rna_id_attribute=False):
    for indx, base in enumerate(query_sequence):
        entry = []
        if rna_id_attribute:
            if base == "A":
                entry = [1, 0, 0, 0, 0, 1]
            elif base == "C":
                entry = [0, 1, 0, 0, 0, 1]
            elif base == "G":
                entry = [0, 0, 1, 0, 0, 1]
            elif base == "T":
                entry = [0, 0, 0, 1, 0, 1]
            elif base == "U":
                entry = [0, 0, 0, 1, 0, 1]
        else:
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


def prepare_ril_seq_melamed():
    # Choose, whether accessibility is used for the IntaRNA computation for ranking and selecting negative targets for
    # each sRNA:
    intarna_acc_ranking = "N"  # "N" means off, "C" means on
    rna_id_attr = True
    remove_weak_edges = True
    edge_weight_threshold = 0.05
    length_threshold = 400
    threads = 52

    # Choose the amount and distribution of sampled non-RIL-Seq negative instances:
    fold_sampling_of_negatives = 20
    # sampling split:
    fold_sampling_top_10 = round(fold_sampling_of_negatives / 2)
    fold_sampling_top_10_to_50 = round(fold_sampling_of_negatives / 4)
    fold_sampling_rest = round(fold_sampling_of_negatives / 4)

    # Define locations for output:
    # file_name_overview_df = "data/test_ril_seq_dir_x_20_10_22/interactions_overview_df.pkl"

    intermediate_saving_file = "data/test_ril_seq_dir_xy_21_11_22/intermediate_interactions_df.pkl"

    # base_name_pickle_output_lists = "data/test_ril_seq_dir_x_20_10_22/raw/ril_seq_melamed_" \
    #                                 "gnn_nested_input_list_"  # (rest of file names is added for each list)
    # file_name_no_fav_intarna_interaction = "data/test_ril_seq_dir_x_20_10_22/no_favorable_interaction_" \
    #                                        "ril_seq_melamed_20_10_22.pkl"
    # pickle_list_size = 2000

    # Load datasets:
    srnas_df, utr_df, benchmark_df = load_benchmark_data()
    id_look_up_df, e_coli_all_genes_table, b_num_up_down_df = get_id_conversion_files()
    genome_file = "data/e_coli_k12_mg1655_genome_NC_000913_2.fasta"
    genome = read_genome(genome_file)
    ril_seq_combined_table = load_ril_seq_melamed(genome, id_look_up_df, e_coli_all_genes_table, b_num_up_down_df)

    ###
    # Create df with positive and negative non-RIL-Seq interactions:

    # create df with the UTRs from the benchmark UTR-file that do not appear in the RIL-Seq dataset:
    idxs = []
    for i in range(len(utr_df)):
        if (utr_df.loc[i]["locus_tag"] == ril_seq_combined_table["rna_1_b_num"]).any():
            idxs = idxs + [i]
        elif (utr_df.loc[i]["locus_tag"] == ril_seq_combined_table["rna_2_b_num"]).any():
            idxs = idxs + [i]

    utr_df_no_ril_seq_overlap = utr_df.drop(idxs).reset_index(drop=True)

    # -> Now UTRs should have the same locus-tags as the according genes. Overlaps should be removed.

    # Create df to collect non-RIL-Seq interactions:
    non_ril_seq_df = pd.DataFrame(columns=['rna_1', 'rna_2', 'rna_1 sequence', 'rna_2 sequence', 'rna_1_b_num',
                                           'rna_2_b_num', 'rna_1_ecocyc_id', 'rna_2_ecocyc_id', 'label'])

    # create df with unique rna_2s from ril_seq_combined_table and their value counts:

    # not all rna_2s in ril_seq_combined_table have locus-tags.
    ril_seq_unique_srna_df = ril_seq_combined_table.value_counts(subset="rna_2")
    ril_seq_unique_srna_df = pd.DataFrame({"rna_2_name": ril_seq_unique_srna_df.index,
                                           "counts": ril_seq_unique_srna_df})

    ril_seq_unique_srna_df.reset_index(drop=True, inplace=True)

    # Remove sequences longer than the length threshold, before running IntRNA (longer sequences take
    # much more time):
    sequences_over_threshold = []
    for i in range(len(ril_seq_unique_srna_df)):
        rna_2_name = ril_seq_unique_srna_df.loc[i]["rna_2_name"]
        rna_2_seq = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2 sequence"]
        if len(rna_2_seq) > length_threshold:
            sequences_over_threshold = sequences_over_threshold + [i]

    ril_seq_unique_srna_df = ril_seq_unique_srna_df.drop(sequences_over_threshold).reset_index(drop=True)

    # Go through (unique) sRNAs in RIL-Seq dataset:
    accessibility = intarna_acc_ranking
    for i in tqdm(range(len(ril_seq_unique_srna_df))):
        rna_2_name = ril_seq_unique_srna_df.loc[i]["rna_2_name"]
        #print(rna_2_name)
        rna_2_counts = ril_seq_unique_srna_df.loc[i]["counts"]
        rna_2_locus_tag = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2_b_num"]
        rna_2_seq = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2 sequence"]
        if not rna_2_locus_tag == "not defined":
            # Go through utr_df_no_ril_seq_overlap: If UTR is in benchmark_df, put interactions in the non-RIL-Seq interactions df
            # (They are positives and should be the ones that are not in RIL-Seq dataset)
            benchmark_subset = benchmark_df[benchmark_df["srna_ltag"] == rna_2_locus_tag]
            benchmark_interactions = []
            for j in range(len(utr_df_no_ril_seq_overlap)):
                rna_1_locus_tag = utr_df_no_ril_seq_overlap.loc[j]["locus_tag"]
                if (benchmark_subset["target_ltag"] == rna_1_locus_tag).any():
                    benchmark_interactions = benchmark_interactions + [j]
                    # add these to non_ril_seq_df, remove from this one

            non_ril_seq_positives_ltags = utr_df_no_ril_seq_overlap.loc[benchmark_interactions]["locus_tag"]
            non_ril_seq_positives_seqs = utr_df_no_ril_seq_overlap.loc[benchmark_interactions]["sequence"]
            non_ril_seq_positives_df = pd.DataFrame({"rna_2": [rna_2_name] * len(non_ril_seq_positives_ltags),
                                                     "rna_1 sequence": non_ril_seq_positives_seqs,
                                                     "rna_2 sequence": [rna_2_seq] * len(non_ril_seq_positives_ltags),
                                                     "rna_1_b_num": non_ril_seq_positives_ltags,
                                                     "rna_2_b_num": [rna_2_locus_tag] * len(
                                                         non_ril_seq_positives_ltags),
                                                     "label": [1] * len(non_ril_seq_positives_ltags)})
            non_ril_seq_df = pd.concat([non_ril_seq_df, non_ril_seq_positives_df])
            utr_no_overlaps = utr_df_no_ril_seq_overlap.drop(benchmark_interactions).reset_index(drop=True)

            # Remove sequences longer than the length threshold, before running IntaRNA (longer sequences take
            # much more time):
            utr_no_overlaps = utr_no_overlaps.drop(
                utr_no_overlaps[utr_no_overlaps["sequence"].map(len) > length_threshold].index)

            sample_df = intarna_ranking(utr_no_overlaps, rna_2_seq, accessibility, rna_2_counts, fold_sampling_top_10,
                                        fold_sampling_top_10_to_50, fold_sampling_rest, threads)

            for utr in sample_df.iterrows():
                rna_1_ltag = utr[1]["id1"]
                rna_1_seq = utr_df_no_ril_seq_overlap[utr_df_no_ril_seq_overlap["locus_tag"] == rna_1_ltag]["sequence"].iloc[0]
                temp_df = pd.DataFrame({"rna_2": [rna_2_name],
                                        "rna_1 sequence": [rna_1_seq],
                                        "rna_2 sequence": [rna_2_seq],
                                        "rna_1_b_num": [rna_1_ltag],
                                        "rna_2_b_num": [rna_2_locus_tag],
                                        "label": [0]})
                non_ril_seq_df = pd.concat([non_ril_seq_df, temp_df])

    non_ril_seq_df = non_ril_seq_df.reset_index(drop=True)

    ###
    # Create df with negative RIL-Seq interactions:

    ril_seq_negatives_df = pd.DataFrame(columns=['rna_1', 'rna_2', 'rna_1 sequence', 'rna_2 sequence', 'rna_1_b_num',
                                                 'rna_2_b_num', 'rna_1_ecocyc_id', 'rna_2_ecocyc_id', 'label'])

    accessibility = intarna_acc_ranking
    for i in tqdm(range(len(ril_seq_unique_srna_df))):
        rna_2_name = ril_seq_unique_srna_df.loc[i]["rna_2_name"]
        #print(rna_2_name)
        rna_2_counts = ril_seq_unique_srna_df.loc[i]["counts"]
        rna_2_locus_tag = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2_b_num"]
        rna_2_seq = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2 sequence"]
        rna_2_ecocyc_id = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name].iloc[0]["rna_2_ecocyc_id"]

        ##
        # use only UTRs from RIL-Seq dataset that are not involved in interactions with this rna_2 (sRNA):
        ril_seq_target_subset_names = ril_seq_combined_table[ril_seq_combined_table["rna_2"] == rna_2_name]
        ril_seq_target_subset_ltags = ril_seq_combined_table[ril_seq_combined_table["rna_2_b_num"] == rna_2_locus_tag]
        ril_seq_target_subset_seq = ril_seq_combined_table[ril_seq_combined_table["rna_2 sequence"] == rna_2_seq]
        ril_seq_target_subset_ecocyc = ril_seq_combined_table[ril_seq_combined_table["rna_2_ecocyc_id"] == rna_2_ecocyc_id]

        ril_seq_unique_rna_1s = ril_seq_combined_table.drop_duplicates(subset="rna_1", keep="first", ignore_index=True)
        ril_seq_unique_rna_1s = ril_seq_unique_rna_1s.drop_duplicates(subset="rna_1_b_num", keep="first",
                                                                      ignore_index=True)
        double = pd.concat([ril_seq_unique_rna_1s, ril_seq_target_subset_names, ril_seq_target_subset_ltags,
                            ril_seq_target_subset_seq, ril_seq_target_subset_ecocyc],
                           ignore_index=True)
        ril_seq_unique_rna_1s_no_overlap = double.drop_duplicates(subset="rna_1", keep=False, ignore_index=True)
        ril_seq_unique_rna_1s_no_overlap = ril_seq_unique_rna_1s_no_overlap.drop_duplicates(subset="rna_1_b_num",
                                                                                            keep=False,
                                                                                            ignore_index=True)
        ril_seq_unique_rna_1s_no_overlap = ril_seq_unique_rna_1s_no_overlap.drop_duplicates(subset="rna_1 sequence",
                                                                                            keep=False,
                                                                                            ignore_index=True)
        ril_seq_unique_rna_1s_no_overlap = ril_seq_unique_rna_1s_no_overlap.drop_duplicates(subset="rna_1_ecocyc_id",
                                                                                            keep=False,
                                                                                            ignore_index=True)
        ##

        possible_negative_utrs = ril_seq_unique_rna_1s_no_overlap[["rna_1_ecocyc_id", "rna_1 sequence"]].reset_index(drop=True)

        # Remove sequences longer than the length threshold, before running IntRNA (longer sequences take
        # much more time):
        possible_negative_utrs = possible_negative_utrs.drop(
            possible_negative_utrs[possible_negative_utrs["rna_1 sequence"].map(len) > length_threshold].index)

        sample_df = intarna_ranking(possible_negative_utrs, rna_2_seq, accessibility, rna_2_counts,
                                    fold_sampling_top_10, fold_sampling_top_10_to_50, fold_sampling_rest, threads)

        sample_df = sample_df.drop_duplicates(subset="id1", keep="first", ignore_index=True)
        for utr in sample_df.iterrows():
            rna_1_ecocyc_id = utr[1]["id1"]
            rna_1_seq = ril_seq_combined_table[ril_seq_combined_table["rna_1_ecocyc_id"] == rna_1_ecocyc_id]["rna_1 sequence"].iloc[0]
            rna_1_ltag = ril_seq_combined_table[ril_seq_combined_table["rna_1_ecocyc_id"] == rna_1_ecocyc_id]["rna_1_b_num"].iloc[0]
            rna_1_name = ril_seq_combined_table[ril_seq_combined_table["rna_1_ecocyc_id"] == rna_1_ecocyc_id]["rna_1"].iloc[0]
            temp_df = pd.DataFrame({"rna_1": [rna_1_name],
                                    "rna_2": [rna_2_name],
                                    "rna_1 sequence": [rna_1_seq],
                                    "rna_2 sequence": [rna_2_seq],
                                    "rna_1_b_num": [rna_1_ltag],
                                    "rna_2_b_num": [rna_2_locus_tag],
                                    "rna_1_ecocyc_id": [rna_1_ecocyc_id],
                                    "rna_2_ecocyc_id": [rna_2_ecocyc_id],
                                    "label": [0]})
            ril_seq_negatives_df = pd.concat([ril_seq_negatives_df, temp_df])

    ril_seq_negatives_df = ril_seq_negatives_df.reset_index(drop=True)
    ril_seq_negatives_df["source"] = "ril_seq"

    non_ril_seq_df["source"] = "non_ril_seq"

    # Combine non-RIL-Seq interactions and positives and negatives from RIL-Seq dataset:
    ril_seq_combined_table_temp = pd.DataFrame({'rna_1': ril_seq_combined_table["rna_1"],
                                                'rna_2': ril_seq_combined_table["rna_2"],
                                                'rna_1 sequence': ril_seq_combined_table["rna_1 sequence"],
                                                'rna_2 sequence': ril_seq_combined_table["rna_2 sequence"],
                                                'rna_1_b_num': ril_seq_combined_table["rna_1_b_num"],
                                                'rna_2_b_num': ril_seq_combined_table["rna_2_b_num"],
                                                'rna_1_ecocyc_id': ril_seq_combined_table["rna_1_ecocyc_id"],
                                                'rna_2_ecocyc_id': ril_seq_combined_table["rna_2_ecocyc_id"],
                                                'label': [1] * len(ril_seq_combined_table),
                                                'source': ["ril_seq"] * len(ril_seq_combined_table)})

    interactions_df = pd.concat([ril_seq_combined_table_temp, ril_seq_negatives_df, non_ril_seq_df])

    # order according to rna_2 names?:
    interactions_df.sort_values("rna_2", inplace=True)

    # remove entries with sequence lengths above a threshold:
    interactions_df = interactions_df.drop(
        interactions_df[interactions_df["rna_1 sequence"].map(len) > length_threshold].index)
    interactions_df = interactions_df.drop(
        interactions_df[interactions_df["rna_2 sequence"].map(len) > length_threshold].index)

    interactions_df = interactions_df.reset_index(drop=True)
    interactions_df["interaction_index"] = interactions_df.index

    # For some 5UTRs and EST5UTRs there are problems with the overlaps. Some that are interacting with the rna_2
    # are not filtered out for creating the negative RIL-seq interactions. The problem are probably missing locus_tags
    # and varying names. This could be solved, by adding the locus_tags when retrieving the sequences or by filtering
    # interacting rna_1s out by ecocyc_ids and sequences as well.
    # The solution for now:
    #interactions_df = interactions_df.drop_duplicates(subset=["rna_1_b_num", "rna_2_b_num"], keep="first",
    #                                                  ignore_index=True)
    # -> should be solved!

    with open(intermediate_saving_file, "wb") as intermediate_out:
        rick.dump(interactions_df, intermediate_out)

    return interactions_df


def prepare_graphs(interactions_df, threads):
    rna_id_attr = True
    remove_weak_edges = True
    edge_weight_threshold = 0.05

    file_name_overview_df = "data/test_ril_seq_dir_xy_21_11_22/interactions_overview_df.pkl"

    base_name_pickle_output_lists = "data/test_ril_seq_dir_xy_21_11_22/raw/ril_seq_melamed_" \
                                    "gnn_nested_input_list_"  # (rest of file names is added for each list)
    file_name_no_fav_intarna_interaction = "data/test_ril_seq_dir_xy_21_11_22/no_favorable_interaction_" \
                                           "ril_seq_melamed_21_11_22.pkl"
    pickle_list_size = 2000

    #return interactions_df, ril_seq_negatives_df, non_ril_seq_df, ril_seq_combined_table

    interactions_df["no_favorable_interaction"] = ""
    intarna_acc_ranking = "N"  # "N" means off, "C" means on (accessibility for IntaRNA)

    accessibility = intarna_acc_ranking
    # Go through the rna_1s of one rna_2 at a time and build graphs:
    unique_rna_2_df = interactions_df.drop_duplicates(subset="rna_2")

    exceptions = 0
    nr_sequences = 0
    nr_overall_sequences = 0
    start_list = 1
    file_name = "place_holder"

    for row in tqdm(unique_rna_2_df.iterrows(), total=unique_rna_2_df.shape[0]):
        rna_2 = row[1]["rna_2"]
        rna_2_seq = row[1]["rna_2 sequence"]

        # Run RNAplfold once for each rna_2:
        plfold_rna_2 = use_plfold(rna_2_seq)

        # Run IntaRNA for chunks of as many targets as threads at a time:
        # (52 threads are possible on the moscow-mule server, 40 still leaves a bit for others, 25 is save.)
        temp_intarna_target_fasta = "temp_intarna_targets.fasta"
        output_file = "multi_intarna_spotprobs"
        output_file_energies = "multi_intarna_energies.csv"
        utr_subset = interactions_df[interactions_df["rna_2"] == rna_2].reset_index(drop=True)

        with open(temp_intarna_target_fasta, "w") as f:
            fasta = ""
            for utr in utr_subset.iterrows():
                entry = ">" + str(utr[1]["interaction_index"]) + "\n" + utr[1]["rna_1 sequence"] + "\n"
                fasta = fasta + entry
            f.write(fasta)

        size_subset = len(utr_subset)
        for i in range(1, size_subset, threads):

            if size_subset - i >= threads:
                target_set = str(i) + "-" + str(i + (threads - 1))
                target_set_len = threads
            else:
                target_set = str(i) + "-" + str(size_subset)
                target_set_len = size_subset - i

            intarna_arguments = "IntaRNA -t " + temp_intarna_target_fasta + " -q " + rna_2_seq + \
                                " --outMode=C" + " --outCsvCols=id1,id2,E" + " --out " + output_file_energies + \
                                " --out spotProb:" + output_file + ".csv" + " --qIdxPos0=0 --tIdxPos0=0" + \
                                " --acc=" + accessibility + " --tset=" + target_set + " --threads=" + str(threads)

            subprocess.run(intarna_arguments, shell=True)

            intarna_energy_df = pd.read_csv(output_file_energies, sep=";")

            # now build each of the graphs:
            for j in range(i, (i + target_set_len)):
                interaction = utr_subset.iloc[j - 1]  # j - 1 because indexing for IntaRNA target set starts at one,
                # for dfs it starts at 0
                interaction_index = interaction["interaction_index"]
                entry_list = []
                index_interactions_df = interactions_df[interactions_df["interaction_index"] == interaction_index].index[0]
                label = interactions_df["label"].iloc[index_interactions_df]
                rna_1_name = interactions_df["rna_1"].iloc[index_interactions_df]
                rna_2_name = interactions_df["rna_2"].iloc[index_interactions_df]
                intarna_energy_entry = intarna_energy_df.loc[intarna_energy_df["id1"] == interaction_index]
                if intarna_energy_entry.empty:
                    no_favorable_interaction = True
                    #interactions_df["no_favorable_interaction"][index_interactions_df] = no_favorable_interaction
                    interactions_df.loc[index_interactions_df, "no_favorable_interaction"] = no_favorable_interaction
                else:
                    no_favorable_interaction = False
                    #interactions_df["no_favorable_interaction"][index_interactions_df] = no_favorable_interaction
                    interactions_df.loc[index_interactions_df, "no_favorable_interaction"] = no_favorable_interaction
                    intarna_energy = intarna_energy_entry["E"].iloc[0]
                    positive_interaction_energy = intarna_energy * (-1)

                    rna_1_seq = interactions_df["rna_1 sequence"].loc[index_interactions_df]

                    # covalent interactions:
                    edge_index, edge_attributes, feature_vector = target_cov_bonds(rna_1_seq, rna_id_attribute=rna_id_attr)
                    edge_index, edge_attributes, feature_vector = query_cov_bonds(rna_2_seq, rna_1_seq, edge_index,
                                                                                  edge_attributes, feature_vector,
                                                                                  rna_id_attribute=rna_id_attr)

                    plfold_rna_1 = use_plfold(rna_1_seq)
                    file_index = j % threads    # the index of the IntaRNA outpu-files always starts at 1 and goes up
                    # for the number of threads.
                    if file_index == 0:
                        file_index = threads
                    specific_spotprob_file = output_file + "-t" + str(file_index) + "q1.csv"
                    edge_index, edge_attributes = plfold_target_in(plfold_rna_1, edge_index, edge_attributes)
                    edge_index, edge_attributes = plfold_query_in(rna_1_seq, plfold_rna_2, edge_index, edge_attributes)
                    edge_index, edge_attributes = intarna_in(edge_index, edge_attributes, specific_spotprob_file)

                    if remove_weak_edges:
                        edge_index_keep = []
                        edge_attributes_keep = []
                        for k, attr in enumerate(edge_attributes):
                            if attr[3] >= edge_weight_threshold:
                                edge_attributes_keep.append(attr)
                                edge_index_keep.append(edge_index[k])
                        edge_index = edge_index_keep
                        edge_attributes = edge_attributes_keep

                    entry_list.append(feature_vector)
                    entry_list.append(edge_index)
                    entry_list.append(edge_attributes)
                    entry_list.append(label)
                    entry_list.append(positive_interaction_energy)
                    entry_list.append([rna_1_name, rna_2_name])
                    entry_list.append(interaction_index)

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
                    #if nr_overall_sequences % 500 == 0:
                        #print(nr_overall_sequences)

    # Save the interactions overview df:
    pickle_out = open(file_name_overview_df, "w+b")
    rick.dump(interactions_df, pickle_out)
    pickle_out.close()

    #return interactions_df, ril_seq_negatives_df, non_ril_seq_df, ril_seq_combined_table


if __name__ == "__main__":
    prepare_ril_seq_melamed()




