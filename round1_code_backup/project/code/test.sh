#测试数据路径
testfile="../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv"

CUDA_VISIBLE_DEVICES=3 python ./test/predict_mutil_dropout_cls_cat_11.py --testfile $testfile &
CUDA_VISIBLE_DEVICES=4 python ./test/predict_mutil_dropout_cls_cat_15.py --testfile $testfile &
CUDA_VISIBLE_DEVICES=5 python ./test/predict_electra_mutil_dropout_cls_cat_16.py --testfile $testfile &
CUDA_VISIBLE_DEVICES=6 python ./test/predict_roberta_mutil_dropout_cls_cat_14.py --testfile $testfile &
CUDA_VISIBLE_DEVICES=7 python ./test/predict_martin_model.py --testfile $testfile &

wait

python ./test/aveage_result.py
zip ../prediction_result/result/result-average.zip ../prediction_result/result/result-average.txt