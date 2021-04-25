testfile="/remote-home/source/tianchi/data/gaiic_track3_round1_testA_20210228.tsv"

CUDA_VISIBLE_DEVICES=0 python predict_mutil_dropout_cls_cat_11.py --testfile $testfile
CUDA_VISIBLE_DEVICES=1 python predict_mutil_dropout_cls_cat_15.py --testfile $testfile
CUDA_VISIBLE_DEVICES=2 python predict_electra_mutil_dropout_cls_cat_16.py --testfile $testfile
CUDA_VISIBLE_DEVICES=3 python predict_roberta_mutil_dropout_cls_cat_14.py --testfile $testfile

python aveage_result.py
zip result-average.zip result-average.txt