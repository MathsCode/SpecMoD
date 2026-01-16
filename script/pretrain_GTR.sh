#!/bin/bash

# 配置路径
TARGET_MODEL="/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B/"
TRAIN_JSON="/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/sharegpt"
SAVE_DIR='/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/global_router_pretrain/'
CONFIG='/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/router_train/pretrain/router_config.json'
DS_CONFIG='/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/router_train/pretrain/ds_config.json'

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# 训练
deepspeed --num_gpus=8 pretrain/train_router.py \
    --target_model_path ${TARGET_MODEL} \
    --train_data_path ${TRAIN_JSON} \
    --savedir ${SAVE_DIR} \
    --config ${CONFIG} \
    --deepspeed_config ${DS_CONFIG}

echo "Training completed! Checkpoints saved to ${SAVE_DIR}"
