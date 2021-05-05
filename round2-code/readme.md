# experment result

model_1 baseline(24layers)
model_2 8layers fpretrian-24 pgd
model_3 12layers pgd
model_4 12 fgm
model_5 24 fgm
model_6 24 pabee


CUDA_VISIBLE_DEVICES=5 python fineturning_macbert_base.py --fold_name ./model_31 --attack_method pgd 

CUDA_VISIBLE_DEVICES=6 python fineturning_macbert_base.py --fold_name ./model_32 --attack_method freelb --epoches 20 --batch_size 256

CUDA_VISIBLE_DEVICES=7 python fineturning_nezha_base.py --fold_name ./model_33 --attack_method freelb-md --epoches 10

CUDA_VISIBLE_DEVICES=4 python fineturning_nezha_base.py --model_type headwithmd --fold_name ./model_34 --attack_method freelb --epoches 10




CUDA_VISIBLE_DEVICES=2 python fineturning_nezha_base_self_ensamble.py --model_type clscat --fold_name ./model_kfold_1 --attack_method freelb --epoches 10

CUDA_VISIBLE_DEVICES=1 python fineturning_nezha_base_self_ensamble.py --model_type clscat --fold_name ./model_kfold_2 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=3 python fineturning_nezha_base_self_ensamble.py --model_type headwithmd --fold_name ./model_kfold_3 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=4 python fineturning_roberta_base.py --model_type clscat --fold_name ./model_roberta_1 --attack_method freelb --epoches 10 --batch_size 256


CUDA_VISIBLE_DEVICES=0 python fineturning_nezha_base.py --fold_name ./nezha_base_v2_4_28_1 --attack_method fgm --epoches 10 --batch_size 256


dev/dev_data/train 没什么用的数据（dev是一部分增强出的正例加上round1的负例， train为round1，round2相互的增强的数据）

train-dual 传递增强数据
train-exchange 交换位置的增强数据


CUDA_VISIBLE_DEVICES=2 python fineturning_nezha_base.py --fold_name ./nezha_base_v2_4_28_5 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=3 python fineturning_nezha_base.py --fold_name ./nezha_base_v2_4_28_6 --attack_method pgd --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=5 python fineturning_macbert_base.py --fold_name ./bert_base_v2_4_28_1 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=6 python fineturning_macbert_base.py --fold_name ./bert_base_v2_4_28_7 --attack_method fgm --epoches 10 --batch_size 256



5.1
CUDA_VISIBLE_DEVICES=4 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_1_2 --attack_method pgd --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=5 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_1_3 --attack_method freelb --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=4 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_1_4 --attack_method fgm --epoches 10 --batch_size 128

CUDA_VISIBLE_DEVICES=4 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_1_4 --attack_method fgm --epoches 10 --batch_size 128


5.2

CUDA_VISIBLE_DEVICES=4 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_2_1 --attack_method fgm --epoches 5 --batch_size 256

CUDA_VISIBLE_DEVICES=5 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_2_2 --attack_method freelb --epoches 5 --batch_size 256

CUDA_VISIBLE_DEVICES=5 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_2_3 --checkpoint nezha_base_output_only_true_v1/checkpoint-50000 --attack_method pgd --epoches 10 --batch_size 256

data path is /remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv, origin model is /remote-home/zyfei/project/tianchi/models/nezha-base-www, and trained model saved /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_2_v3_clean_round2data, data cache is /remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_cleandata, /remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_round2data, tokenizer is /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30_v2_round2data

5.3

CUDA_VISIBLE_DEVICES=6 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_3_1 --checkpoint nezha_base_output_only_true_v1/checkpoint-50000 --attack_method pgd --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=7 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_3_2 --checkpoint nezha_base_output_only_true_v1/checkpoint-50000 --attack_method fgm --epoches 10 --batch_size 256

/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_3_clean_round2data
CUDA_VISIBLE_DEVICES=3 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_4_1 --checkpoint nezha_base_output_5_3_clean_round2data --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=6 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_4_1 --checkpoint nezha_base_output_5_3_clean_round2data --attack_method pgd --epoches 10 --batch_size 128

data path is /remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv, origin model is /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_3_clean_round2data, and trained model saved /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_3_clean_round2data_2, data cache is /remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_cleandata, /remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_round2data, tokenizer is /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_2_v3_clean_round2data/checkpoint-20000

CUDA_VISIBLE_DEVICES=3 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_4_3 --checkpoint nezha_base_output_5_3_clean_round2data_2/checkpoint-10000 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=6 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_4_2 --checkpoint nezha_base_output_5_3_clean_round2data_2/checkpoint-30000 --attack_method pgd --epoches 10 --batch_size 128


CUDA_VISIBLE_DEVICES=0 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_5_1 --checkpoint nezha_base_output_5_3_clean_round2data_3 --attack_method pgd --epoches 10 --batch_size 256


CUDA_VISIBLE_DEVICES=1 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_5_2 --checkpoint nezha_base_output_5_3_clean_round2data_3 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=2 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_5_2 --checkpoint nezha_base_output_5_3_clean_round2data_3 --attack_method mutildrop --epoches 5 --batch_size 256

CUDA_VISIBLE_DEVICES=3 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_5_4 --checkpoint nezha_base_output_5_3_clean_round2data_3 --attack_method fgm --epoches 10 --batch_size 256

CUDA_VISIBLE_DEVICES=2 python fineturning_nezha_base_k_fold.py --fold_name ./nezha_5_5_5 --checkpoint nezha_base_output_5_3_clean_round2data_3 --attack_method fgm --epoches 10 --batch_size 256 --classifier_dropout 0.3


2.1 v1 origin_model nezha_base_output_4_30_v2_round2/checkpoint-20000
further pretrain param lr 5e-5 epoches 100 per_batch_size 128 
kfold 3 epochs = 10 lr = 1e-5 batch_size = 128 model_type = "clscat" attack_method = "fgm" n_split = 3 onnx_path = "../onnx_models"
