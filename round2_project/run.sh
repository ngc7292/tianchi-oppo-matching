CUDA_VISIBLE_DEVICES=0,1,2,3 python ./code/further_pretrain_nezha.py
CUDA_VISIBLE_DEVICES=0 python ./code/fine_turning_nezha_base.py
CUDA_VISIBLE_DEVICES=0 python ./code/convert.py
CUDA_VISIBLE_DEVICES=0 python ./code/predict.py


