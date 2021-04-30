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
