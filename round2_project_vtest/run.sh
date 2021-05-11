cd code

#CUDA_VISIBLE_DEVICES=0,1,2,3 python further_pretrain_nezha.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 python further_pretrain_roberta.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 python further_pretrain_bert.py
#
#wait
CUDA_VISIBLE_DEVICES=0 python fineturning.py

CUDA_VISIBLE_DEVICES=0 python predict.py


