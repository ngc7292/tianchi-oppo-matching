# -*- coding: utf-8 -*-
"""
__title__="aveage_result"
__author__="ngc7293"
__mtime__="2021/4/7"
"""
def load_result(result_path):
    r = []
    with open(result_path, "r") as f:
        data = f.readlines()
        for i in data:
            r.append(float(i.replace("\n", "")))
    return r


r_1 = load_result("../prediction_result/result-11.txt")
r_2 = load_result("../prediction_result/result-14.txt")
r_3 = load_result("../prediction_result/result-15.txt")
r_4 = load_result("../prediction_result/result-16.txt")
r_5 = load_result("../prediction_result/result-du.txt")

import numpy as np

r_1 = np.expand_dims(r_1, axis=-1)
r_2 = np.expand_dims(r_2, axis=-1)
r_3 = np.expand_dims(r_3, axis=-1)
r_4 = np.expand_dims(r_4, axis=-1)
r_5 = np.expand_dims(r_5, axis=-1)

result_e = np.concatenate([r_1, r_2, r_3, r_4,r_5], axis=-1)

result_e_avg = np.average(result_e, axis=-1)

with open("../prediction_result/result-average.txt", "w") as f:
    for i in result_e_avg:
        f.write(str(i) + "\n")
