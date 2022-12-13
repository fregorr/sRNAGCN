import numpy as np
import subprocess
import pandas as pd
import pickle as rick
import csv
import argparse
import copy


# Einlesen einer CSV Tabelle mit den vorbereiteten Daten (Sequenzvarianten und die Häufigkeiten, mit denen sie in den
# verschiedenen Bins vertreten sind)
# Output: pandas-Dataframe
def read_file(file_name):
    df = pd.read_csv(file_name, sep=";")
    return df


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
        feature_vector.append(entry)
        if indx <= (len(query_sequence) - 2):
            index = (len(target_sequence) - 1) + 1 + indx
            edge_forward = [index, (index + 1)]
            edge_backwards = [(index + 1), index]
            edge_index.append(edge_forward)
            attribute = [1, 0, 0, 1]
            edge_attributes.append(attribute)
            edge_index.append(edge_backwards)
            edge_attributes.append(attribute)     # Attribut muss sowohl für edge_forward, als auch für edge_backwards angefügt werden.
    return edge_index, edge_attributes, feature_vector


# Ruft und verwendet RNAplfold auf die eingegebene Sequenz. Ein Cutoff von 0.001 würde bedeuten, dass alle Basenpaar-Wahr-
# scheinlichkeiten die darüber liegen angezeigt werden würden.
# Input: Sequenz als String
# Output: Wurzeln der Basenpaar-Wahrscheinlichkeiten mit Indexpositionen als String. (Index startet bei 1, statt wie
# bei Python mit 0. Dies wird später korrigiert)
def use_plfold(sequence):
    plfold_arguments = "echo " + sequence  + " | RNAplfold"
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
        plfold_index_1 = int(entry[0]) -1
        plfold_index_2 = int(entry[1]) -1
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


def use_intarna(target_sequence, query_sequence, output_file):
    intarna_arguments = "IntaRNA -t " + target_sequence + " -q " + query_sequence + " --out spotProb:" + output_file + " --qIdxPos0=0 --tIdxPos0=0"
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


def make_label(list_frequencies_in_bins, list_bin_weights, list_freq_sums_bins):
    # Normalisieren:
    zaehler = 0
    for i in range(0, 5):
        bin = i + 1
        x = bin * list_bin_weights[i] * list_frequencies_in_bins[i] / list_freq_sums_bins[i]
        zaehler = zaehler + x
    nenner = 0
    for i in range(0, 5):
        y = list_frequencies_in_bins[i] * list_bin_weights[i] / list_freq_sums_bins[i]
        nenner = nenner + y
    label_bin = zaehler/nenner
    return label_bin


# Start:

# Eingabe:
ptsg = "TAATAAATAAAGGGCGCTTAGATGCCCTGTACACGGCGAGGCTCTCCCCCCTTGCCACGCGTGA" \
                  "GAACGTAAAAAAAGCACCCATACTCAGGAGCACTCTCAATTATGTTTAAGAATGCATTTGCTAACCTGCAA"
#query_file = input("Please enter the file name of the CSV file with the pre-processed rna-seq reads of the sRNA: \n")
#df_reads = read_file(query_file)


description = "Takes the pre-processed data from Poddar et al., calls RNAplfold and " \
             "IntaRNA on it, puts it in the right format for Pytorch geometric and saves " \
             "it as a row of pickle files"

#parser = argparse.ArgumentParser(description=description)

#parser.add_argument('--t', metavar='target_sequence', type=str, default=ptsg, help='')
#parser.add_argument('query_file', metavar='query_file', type=str, help='Enter name of CSV-file')
#parser.add_argument('--s', metavar='pickle_list_size', type=int, default=1000, help='Define the size of output lists')

#args = parser.parse_args()

#target_sequence = args.t
#query_file = args.query_file
#pickle_list_size = args.s

target_sequence = ptsg
query_file = "table_reads_per_bin.csv"
pickle_list_size = 2000

df_reads = read_file(query_file)

df_length = len(df_reads)
num_lists = 0
while df_length >= 1:
    num_lists = num_lists + 1
    df_length = df_length - pickle_list_size

print(f"Number of pickle lists: {num_lists}")

# Liste der Prozent aller Zellen, die mit FACS jeweils in ein Bin sortiert wurden (Für dieses Experiment spezifisch):
list_bin_weights = [0.1874, 0.3376, 0.3091, 0.1383, 0.0276]

# Berechnung der Summe aller Reads in den jeweiligen Bins (Je beide Replikate kombiniert):
sum_bin1 = df_reads["bin1 rep1"].sum() + df_reads["bin1 rep2"].sum()
sum_bin2 = df_reads["bin2 rep1"].sum() + df_reads["bin2 rep2"].sum()
sum_bin3 = df_reads["bin3 rep1"].sum() + df_reads["bin3 rep2"].sum()
sum_bin4 = df_reads["bin4 rep1"].sum() + df_reads["bin4 rep2"].sum()
sum_bin5 = df_reads["bin5 rep1"].sum() + df_reads["bin5 rep2"].sum()
list_freq_sums_bins = [sum_bin1, sum_bin2, sum_bin3, sum_bin4, sum_bin5]

edge_index_init = []
edge_attributes_init = []
feature_vector_init = []
# edge_index, edge_attributes und feature_vector für die Zielsequenz (mRNA) und die kovalenten Bindungen.
# Muss nur einmal erstellt werden, da die Zielsequenz gleich bleibt. Für die Such-Sequenzen wird später alles immer
# wieder neu an diese Listen angefügt.
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
    feature_vector_init.append(entry)
    if indx <= (len(target_sequence) - 2): # Because of the indexing from 0, indx (and indx + 1)
        # should never be larger than len(sequence) - 1!
        edge_forward = [indx, (indx + 1)]
        edge_backwards = [(indx + 1), indx]
        edge_index_init.append(edge_forward)
        attribute = [1, 0, 0, 1]
        edge_attributes_init.append(attribute)
        edge_index_init.append(edge_backwards)
        edge_attributes_init.append(attribute)              # Attribut muss sowohl für edge_forward, als auch für edge_backwards angefügt werden.


# Für die Zielsequenz muss RNAplfold auch nur einmal durchgeführt werden:
plfold_target = use_plfold(target_sequence)

# Ruft RNAplfold und IntaRNA für jede sRNA Sequenz-Variante, berechnet das Label und trägt alles zusammen als Liste als Eintrag
# in die gnn_input_list ein.
o = 0
length_dataframe = range(0, len(df_reads))

start_list = 1
nr_sequences = 0
only_one_read = 0
no_fav_interact = 0

for i in length_dataframe:
    edge_index = copy.deepcopy(edge_index_init)
    edge_attributes = copy.deepcopy(edge_attributes_init)
    feature_vector = copy.deepcopy(feature_vector_init)
    #print(f"feature vector init: {feature_vector_init[:10]}")   # debugging
    #print(f"feature vector: {feature_vector[:10]}")   # debugging
    entry_list = []
    gnn_input_list = []
    no_favorable_interactions = []
    entry = df_reads.loc[i]
    query_sequence = entry["sequence"]
    bin1 = entry["bin1 rep1"] + entry["bin1 rep2"]
    bin2 = entry["bin2 rep1"] + entry["bin2 rep2"]
    bin3 = entry["bin3 rep1"] + entry["bin3 rep2"]
    bin4 = entry["bin4 rep1"] + entry["bin4 rep2"]
    bin5 = entry["bin5 rep1"] + entry["bin5 rep2"]
    list_frequencies_in_bins = [bin1, bin2, bin3, bin4, bin5]
    sum_of_reads = sum(list_frequencies_in_bins)
    if sum_of_reads > 2:
        output_file = "intarna_spotprob.csv"
        label = make_label(list_frequencies_in_bins, list_bin_weights, list_freq_sums_bins)
        intarna_stdout = use_intarna(target_sequence, query_sequence, output_file)
        if "nno favorable interaction" in intarna_stdout:
            no_favorable_interactions.append(query_sequence)
            pickle_out = open("no_favorable_interactions.pkl", "a+b")
            rick.dump(no_favorable_interactions, pickle_out)
            pickle_out.close()
            print("No favorable interaction")
            no_fav_interact = no_fav_interact + 1
        else:
            nr_sequences = nr_sequences + 1
            edge_index, edge_attributes, feature_vector = query_cov_bonds(query_sequence, target_sequence, edge_index,
                                                                          edge_attributes, feature_vector)
            find_intarna_energy = intarna_stdout.find("interaction energy = ")
            start_pos_energy = find_intarna_energy + len("interaction energy = ")
            end_pos_energy = intarna_stdout.find(" kcal/mol")
            interaction_energy = float(intarna_stdout[start_pos_energy:end_pos_energy])
            #print(intarna_stdout)   # debugging
            #print(interaction_energy)   # debugging
            positive_interaction_energy = interaction_energy * (-1)
            #print(positive_interaction_energy)      # debugging
            #for node_features in feature_vector:
            #    node_features.append(positive_interaction_energy)
            plfold_query = use_plfold(query_sequence)
            edge_index, edge_attributes = plfold_target_in(plfold_target, edge_index, edge_attributes)
            edge_index, edge_attributes = plfold_query_in(target_sequence, plfold_query, edge_index, edge_attributes)
            edge_index, edge_attributes = intarna_in(edge_index, edge_attributes, output_file)
            #label = make_label(list_frequencies_in_bins, list_bin_weights, list_freq_sums_bins)
            #print(f"new feature vector: {feature_vector[:10]}")  # debugging
            entry_list.append(feature_vector)
            entry_list.append(edge_index)
            entry_list.append(edge_attributes)
            entry_list.append(label)
            entry_list.append(positive_interaction_energy)
            gnn_input_list.append(entry_list)    # Vermutlich unnötig verschachtelt! gnn-input_list könnte man weglassen.
            if nr_sequences == 1:
                file_name = f'energy_gnn_nested_input_list_{start_list}_.pkl'
                start_list = start_list + pickle_list_size
            if nr_sequences == pickle_list_size:
                nr_sequences = 0
            pickle_out = open(file_name, "a+b")
            rick.dump(gnn_input_list, pickle_out)
            pickle_out.close()
            o = o + 1
            print(o)
    else:
        print("only one read")
        only_one_read = only_one_read + 1


print("done")

print(f"Number of usable graphs: {o}, \n"
      f"Number of sequences with only one read: {only_one_read}, \n"
      f"Number with no favorable IntaRNA interaction: {no_fav_interact}")

# So würde man es in einem anderen Skript wieder laden:

# pickle_in = open("gnn_nested_input_list.pkl", "rb")
# gnn_input_list = rick.load(pickle_in)
# pickle_in.close()
