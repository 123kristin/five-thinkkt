#!/bin/bash

# 定义数据集列表和对应的显卡ID
DATASETS=("DBE_KT22" "XES3G5M" "Eedi")
GPUS=(1 2 3)

# 循环遍历数据集并启动训练任务
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    GPU_ID=${GPUS[$i]}
    
    echo "Starting training for $DATASET on GPU $GPU_ID..."
    
    # 设置环境变量 CURRENT_GPU_ID
    export CURRENT_GPU_ID=$GPU_ID
    
    # 启动训练脚本
    # --use_cot 1: 开启 CoT
    # --fold 0: 只训练第 0 折
    # --resume: 开启断点续传
    nohup python scripts_training2testing/examples/wandb_qqkt_train.py \
        --dataset_name $DATASET \
        --use_cot 1 \
        --fold 0 \
        --resume \
        > "logs/qqkt_cot_${DATASET}_gpu${GPU_ID}.log" 2>&1 &
        
    echo "Task for $DATASET launched. Log: logs/qqkt_cot_${DATASET}_gpu${GPU_ID}.log"
done

echo "All tasks launched."
