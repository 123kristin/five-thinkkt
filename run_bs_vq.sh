#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting VCRKT V&Q experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 构造相对保存路径
    REL_SAVE_DIR="saved_model/bs/vq"
    mkdir -p "$REL_SAVE_DIR"
    
    # 构造日志文件名
    LOG_FILE="saved_model/bs/logs/vq_${DATASET}.log"
    
    echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=V&Q"
    echo "[$(date)] Starting Training..." > "$LOG_FILE"
    
    (
        # 切换到脚本目录执行 (保持相对路径一致性)
        cd scripts_training2testing/examples && \
        # 1. 训练
        python wandb_vcrkt_train.py \
        --dataset_name "$DATASET" \
        --model_name "vcrkt" \
        --question_rep_type "v&q" \
        --save_dir "../../$REL_SAVE_DIR" \
        --dim_qc 200 \
        --d_question 1024 \
        --gpu_id "$GPU_ID" \
        --num_epochs 200 \
        --use_wandb 0 \
            >> "../../$LOG_FILE" 2>&1
        
        train_exit_code=$?
        
        if [ $train_exit_code -eq 0 ]; then
            echo "[$(date)] Training process finished." >> "../../$LOG_FILE"
        else
            echo "[$(date)] Training Failed with code $train_exit_code" >> "../../$LOG_FILE"
        fi
    )
    
    echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=V&Q"
}

# 并行运行三个数据集
# 注意：数据集名称必须与 data_config.json 中的键完全匹配 (区分大小写)

# GPU 0: XES3G5M
run_dataset_experiments "XES3G5M" 0 &

# GPU 1: DBE_KT22
run_dataset_experiments "DBE_KT22" 1 &

# GPU 2: nips_task34
run_dataset_experiments "nips_task34" 2 &

# 等待所有后台任务完成
echo "All V&Q experiments launched in parallel. Waiting for completion..."
wait
echo "All V&Q experiments finished."
