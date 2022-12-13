

import subprocess
import regex as re
import numpy as np
import pandas as pd

# Sequenzen, die abgeschnitten werden sollen, können in ein FASTA-File geschrieben werden und an "--adapter_fasta" übergeben werden.
# (müssen länger als 6 bp sein)
# "fastp -i " + input_file + " -o " + output_file + " --length_required " + " --length_limit " + " --adapter_fasta"

# deduplication deaktivieren und mit eigenem Skript machen? (-> Sollte per default inaktiv sein)
# (Ev.: "--dont_eval_duplication" um Duplikation nicht zu bewerten. Wahrsch. auch über Skript)

# Ruft fastp und nutzt es, um Sequenzen mit schlechter Qualität und einer Länge, die anders ist, als genau 150 bp,  rauszufiltern.
# Input: Dateiname fastq-file und ein (möglichst aussagekräftiger) Header als String ( Bsp: >bin1_rep1)
def use_fastp(input_file, header):
    output_file = header[1:] + ".txt"
    arguments_fastp = "fastp -i " + input_file + " -o " + output_file + " --length_required " + "150" + " --length_limit " + "150" + " --disable_adapter_trimming"
    subprocess.run(arguments_fastp, shell=True)
    fastp_data = ""
    with open(output_file, "r") as fastp_file:
        for line in fastp_file:
            fastp_data = fastp_data + line
    return fastp_data

# Nimmt FastQ-File, extrahiert die Sequenzen, schneidet sie auf den Teil zurecht, der später benötigt wird, schreibt sie in fasta-Format und gibt diese als Liste aus.
# Input: FastQ String, Header (Bin und Replikat wie FastA-Header: ">binXrepX")
# Output: Liste mit Sequenzen.
def fastp_to_fasta(fastp_data, header):
    entry_character = re.compile("(?:@SRR)")     # Nur auf @ gehen funktioniert nicht, weils auch im Qualitätsstring auftauchen kann.
    fastp_data = re.split(entry_character, fastp_data)
    i = 0
    for entry in fastp_data:
        try:
            fastp_data[i] = fastp_data[i].split("\n")
            seq = fastp_data[i][1]
            seq = seq[22:101]                                  # Sollte jetzt Position 149 - 227 der SgrS Originalsequenz entsprechen
            fastp_data[i] = header + "\n" + seq
        except:
            fastp_data[i] = ""
        i = i + 1
    fastp_data.remove("")
    fasta_data = fastp_data
    return fasta_data

# Nimmt einer Liste mit Fasta-Sequenzen und generiert eine Liste mit den einzigartigen Sequenzen daraus und ein Dictionary, in das geschrieben wird, wie
#  oft die Sequenzen jeweils vorkommen.
# Input: Liste mit Fasta-Sequenzen
# Output: Liste mit erstem Element: Liste der einzigartigen Sequenzen und zweitem Element: Dictionary mit dem Vorkommen der Sequenzen
def clean_fasta(fasta_data):
    liste = []
    dictionary = {}
    for entry in fasta_data:
        x = entry.split("\n")
        x = x[1]
        if x in liste:
            dictionary[x] = dictionary[x] + 1
            print("old")    # fürs Testen
        else:
            liste.append(x)
            dictionary[x] = 1
            print("new")   # fürs Testen
    list_and_dict = [liste, dictionary]
    return list_and_dict

# Verwendet nacheinander die oben definierten Funktionen.
def clean_data(fastq_file, bin, replicate):
    header = ">" + "bin" + bin + "_rep" + replicate
    fastp_data = use_fastp(fastq_file, header)
    fasta_data = fastp_to_fasta(fastp_data, header)
    list_and_dictionary = clean_fasta(fasta_data)
    return list_and_dictionary


# Start
fastq_file1 = input("Please enter fastq file name of bin 1/ replicate 1: \n")
fastq_file2 = input("Please enter fastq file name of bin 1/ replicate 2: \n")
fastq_file3 = input("Please enter fastq file name of bin 2/ replicate 1: \n")
fastq_file4 = input("Please enter fastq file name of bin 2/ replicate 2: \n")
fastq_file5 = input("Please enter fastq file name of bin 3/ replicate 1: \n")
fastq_file6 = input("Please enter fastq file name of bin 3/ replicate 2: \n")
fastq_file7 = input("Please enter fastq file name of bin 4/ replicate 1: \n")
fastq_file8 = input("Please enter fastq file name of bin 4/ replicate 2: \n")
fastq_file9 = input("Please enter fastq file name of bin 5/ replicate 1: \n")
fastq_file10 = input("Please enter fastq file name of bin 5/ replicate 2: \n")

bin1 = str(1)
bin2 = str(1)
bin3 = str(2)
bin4 = str(2)
bin5 = str(3)
bin6 = str(3)
bin7 = str(4)
bin8 = str(4)
bin9 = str(5)
bin10 = str(5)

rep1 = str(1)
rep2 = str(2)
rep3 = str(1)
rep4 = str(2)
rep5 = str(1)
rep6 = str(2)
rep7 = str(1)
rep8 = str(2)
rep9 = str(1)
rep10 = str(2)

list_and_dict1 = clean_data(fastq_file1, bin1, rep1)
list_and_dict2 = clean_data(fastq_file2, bin2, rep2)
list_and_dict3 = clean_data(fastq_file3, bin3, rep3)
list_and_dict4 = clean_data(fastq_file4, bin4, rep4)
list_and_dict5 = clean_data(fastq_file5, bin5, rep5)
list_and_dict6 = clean_data(fastq_file6, bin6, rep6)
list_and_dict7 = clean_data(fastq_file7, bin7, rep7)
list_and_dict8 = clean_data(fastq_file8, bin8, rep8)
list_and_dict9 = clean_data(fastq_file9, bin9, rep9)
list_and_dict10 = clean_data(fastq_file10, bin10, rep10)

list1 = list_and_dict1[0]
list2 = list_and_dict2[0]
list3 = list_and_dict3[0]
list4 = list_and_dict4[0]
list5 = list_and_dict5[0]
list6 = list_and_dict6[0]
list7 = list_and_dict7[0]
list8 = list_and_dict8[0]
list9 = list_and_dict9[0]
list10 = list_and_dict10[0]


list_all_bins = list1 + list2 + list3 + list4 + list5 + list7 + list8 + list9 + list10                     # Kombiniert alle Listen mit uniquen Sequenzen aus Bins in eine Liste.
array_all_bins = np.array(list_all_bins)                                                                   # Nur kurz in Array umwandeln, damit unique funktioniert und alle doppelten Sequenzen aus der Liste geschmissen werden.
array_all_bins = np.unique(array_all_bins)


bin1rep1 = []
bin1rep2 = []
bin2rep1 = []
bin2rep2 = []
bin3rep1 = []
bin3rep2 = []
bin4rep1 = []
bin4rep2 = []
bin5rep1 = []
bin5rep2 = []
for seq in array_all_bins:
    if seq in list_and_dict1[1]:
        x = list_and_dict1[1][seq]
        bin1rep1.append(x)
    else:
        bin1rep1.append(0)
    if seq in list_and_dict2[1]:
        x = list_and_dict2[1][seq]
        bin1rep2.append(x)
    else:
        bin1rep2.append(0)
    if seq in list_and_dict3[1]:
        x = list_and_dict3[1][seq]
        bin2rep1.append(x)
    else:
        bin2rep1.append(0)
    if seq in list_and_dict4[1]:
        x = list_and_dict4[1][seq]
        bin2rep2.append(x)
    else:
        bin2rep2.append(0)
    if seq in list_and_dict5[1]:
        x = list_and_dict5[1][seq]
        bin3rep1.append(x)
    else:
        bin3rep1.append(0)
    if seq in list_and_dict6[1]:
        x = list_and_dict6[1][seq]
        bin3rep2.append(x)
    else:
        bin3rep2.append(0)
    if seq in list_and_dict7[1]:
        x = list_and_dict7[1][seq]
        bin4rep1.append(x)
    else:
        bin4rep1.append(0)
    if seq in list_and_dict8[1]:
        x = list_and_dict8[1][seq]
        bin4rep2.append(x)
    else:
        bin4rep2.append(0)
    if seq in list_and_dict9[1]:
        x = list_and_dict9[1][seq]
        bin5rep1.append(x)
    else:
        bin5rep1.append(0)
    if seq in list_and_dict10[1]:
        x = list_and_dict10[1][seq]
        bin5rep2.append(x)
    else:
        bin5rep2.append(0)

data = {"sequence": array_all_bins,
        "bin1 rep1": bin1rep1,
        "bin1 rep2": bin1rep2,
        "bin2 rep1": bin2rep1,
        "bin2 rep2": bin2rep2,
        "bin3 rep1": bin3rep1,
        "bin3 rep2": bin3rep2,
        "bin4 rep1": bin4rep1,
        "bin4 rep2": bin4rep2,
        "bin5 rep1": bin5rep1,
        "bin5 rep2": bin5rep2}

unique_seq_bins_df = pd.DataFrame(data,
                                  columns=["sequence", "bin1 rep1", "bin1 rep2", "bin2 rep1",
                                           "bin2 rep2", "bin3 rep1", "bin3 rep2", "bin4 rep1",
                                           "bin4 rep2", "bin5 rep1", "bin5 rep2"])

unique_seq_bins_df.to_csv("table_reads_per_bin.csv", sep=";")

print("done")
