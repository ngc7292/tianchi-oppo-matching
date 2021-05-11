CUDA_VISIBLE_DEVICES=5,6,7 python further-pretrain.py \
    --dataset_name /remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv \
    --model_name_or_path /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_3_clean_round2data \
    --output_dir ./nezha_5_4_1 \
    --weight_decay 0.01 \
    --num_train_epochs 100 \
    --seed 42 \
    --per_device_train_batch_size 128

CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --config ~/.cache/huggingface/accelerate/default_config.yaml python further-pretrain.py \
    --dataset_name /remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv \
    --model_name_or_path /remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_3_clean_round2data \
    --output_dir ./nezha_5_4_1 \
    --weight_decay 0.01 \
    --num_train_epochs 100 \
    --seed 42 \
    --per_device_train_batch_size 128

CUDA_VISIBLE_DEVICES=5,6,7 accelerate test --config ~/.cache/huggingface/accelerate/default_config.yaml