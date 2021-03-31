# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/3/18"
"""
import os
import random
import numpy as np
import torch
import tqdm
import time

from fineturn import load_data, ProjectConfig, MatchingModel


args = ProjectConfig().get_arg()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print(f"load test data...")

data_bundle = load_data(_refresh=False)

print(data_bundle)
data_bundle.datasets['test'].print_field_meta()

model_path = "/remote-home/zyfei/project/tianchi/baseline_nezha/output"
model_name = "best_DataParallel_auc_2021-03-22-01-29-50-873742"

model = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cuda:0'))

model.eval()
start_time = time.time()
res = []
with open("/remote-home/zyfei/project/tianchi/baseline/result/baseline_result_4.txt", 'w') as f:
    for data in tqdm.tqdm(data_bundle.datasets['test']):
        words = torch.tensor([data['words']], dtype=torch.long, device=torch.device('cuda:0'))
        pred = model.predict(words=words)['pred']
        # print(pred.cpu().detach()[0])
        a = float(pred.cpu().detach()[0][1])
        f.write(str(a) + "\n")

print("using time: "+ str(time.time()-start_time))
print("end")

