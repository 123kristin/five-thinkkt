#!/bin/bash

# GPU 1 Tasks: DBE_KT22 and nips_task34
echo "Launching tasks on GPU 1..."

# DBE_KT22
nohup python scripts_training2testing/examples/wandb_qqkt_train.py \
    --dataset_name DBE_KT22 \
    --use_cot 1 \
    --fold 0 \
    --resume \
    > "logs/qqkt_cot_DBE_KT22_gpu1.log" 2>&1 &
echo "Started DBE_KT22 on GPU 1 (PID: $!)"

# nips_task34
# 注意：这里我们使用环境变量显式传递 GPU ID 给 python 进程，虽然 nohup 会继承，但显式指定更安全
CURRENT_GPU_ID=1 nohup python scripts_training2testing/examples/wandb_qqkt_train.py \
    --dataset_name nips_task34 \
    --use_cot 1 \
    --fold 0 \
    --resume \
    > "logs/qqkt_cot_nips_task34_gpu1.log" 2>&1 &
echo "Started nips_task34 on GPU 1 (PID: $!)"


# GPU 3 Tasks: XES3G5M
echo "Launching tasks on GPU 3..."

# XES3G5M
CURRENT_GPU_ID=3 nohup python scripts_training2testing/examples/wandb_qqkt_train.py \
    --dataset_name XES3G5M \
    --use_cot 1 \
    --fold 0 \
    --resume \
    > "logs/qqkt_cot_XES3G5M_gpu3.log" 2>&1 &
echo "Started XES3G5M on GPU 3 (PID: $!)"

echo "All training tasks launched in background."
