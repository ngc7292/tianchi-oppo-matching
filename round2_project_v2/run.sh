cd code

CUDA_VISIBLE_DEVICES=5,6,7 python further_pretrain_nezha.py
CUDA_VISIBLE_DEVICES=5,6,7 python further_pretrain_roberta.py
CUDA_VISIBLE_DEVICES=5,6,7 python further_pretrain_electra.py

CUDA_VISIBLE_DEVICES=5 python fineturning.py
CUDA_VISIBLE_DEVICES=5 python predict.py


