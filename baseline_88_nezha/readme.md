# experience result

pretrain model: nezha-large
pretrain data: enhance data with closure
pretrain strategy: 2-gram

CUDA_VISIBLE_DEVICES=0,1,2,3 python futher_pertrain_n_gram.py 10h

