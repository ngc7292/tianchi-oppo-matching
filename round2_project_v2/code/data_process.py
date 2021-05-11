import os


enhance_data = []
origin_path = "../tcdata/gaiic_track3_round2_train_20210407.tsv"
enhance_path = "../data/enhance.tsv"

with open(origin_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        temp = line.split('\t')
        enhance_data.append("\t".join([temp[0], temp[1], temp[2]]))
        enhance_data.append("\t".join([temp[1], temp[0], temp[2]]))

with open(enhance_path, "w", encoding="utf-8") as f:
    for line in enhance_data:
        f.write(line)
        f.write("\n")