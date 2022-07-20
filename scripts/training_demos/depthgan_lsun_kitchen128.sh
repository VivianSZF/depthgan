#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training DepthGAN on LSUN-Kitchen-128."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/LSUN/kitchen_train_rgbd.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2
TRAIN_ANNO_PATH=$3
VAL_ANNO_PATH=$4

./scripts/dist_train.sh ${GPUS} depthgan \
    --job_name='depthgan_lsun_kitchen128' \
    --seed=0 \
    --resolution=128 \
    --train_dataset=${DATASET} \
    --train_data_file_format='zip' \
    --train_anno_path=${TRAIN_ANNO_PATH} \
    --val_dataset=${DATASET} \
    --val_data_file_format='zip' \
    --val_anno_path=${VAL_ANNO_PATH} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=8 \
    --train_data_mirror=false \
    --data_loader_type='iter' \
    --data_repeat=1 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_init_res=4 \
    --latent_dim_rgb=512 \
    --latent_dim_depth=512 \
    --d_fmaps_factor=0.5 \
    --g_fmaps_factor=0.5 \
    --d_mbstd_groups=8 \
    --g_num_mappings=8 \
    --d_lr=0.0015 \
    --g_lr=0.0015 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_interval=16 \
    --r1_gamma=0.3 \
    --pl_interval=4 \
    --pl_weight=0.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=20000 \
    --g_ema_rampup=0.0 \
    --eval_at_start=false \
    --eval_interval=3200 \
    --ckpt_interval=3200 \
    --log_interval=64 \
    --enable_amp=false \
    --use_ada=false \
    --num_fp16_res=4 \
    --drotloss=50 \
    --rgbrotloss=0.3 \
    --ddloss=0.8 \
    --gdloss=0.001 \
    --keep_ckpt_num=-1 \
    ${@:5}
