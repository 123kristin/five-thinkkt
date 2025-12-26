#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting VCRKT QID experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 构造相对保存路径
    REL_SAVE_DIR="saved_model/bs/qid"
    mkdir -p "$REL_SAVE_DIR"
    
    # 遍历 5 折 (串行)
    for FOLD in 0 1 2 3 4; do
        # 构造日志文件名
        LOG_FILE="saved_model/bs/logs/qid_${DATASET}_fold${FOLD}.log"
        
        echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=QID Fold=$FOLD"
        echo "[$(date)] Starting Training Fold $FOLD..." > "$LOG_FILE"
        
        (
            # 切换到脚本目录执行 (保持相对路径一致性)
            cd scripts_training2testing/examples && \
            # 1. 训练
            python wandb_vcrkt_train.py \
            --dataset_name "$DATASET" \
            --model_name "vcrkt" \
            --question_rep_type "qid" \
            --fold "$FOLD" \
            --save_dir "../../$REL_SAVE_DIR" \
            --dim_qc 200 \
            --gpu_id "$GPU_ID" \
            --num_epochs 200 \
            --use_wandb 0 \
                >> "../../$LOG_FILE" 2>&1
            
            train_exit_code=$?
            
            if [ $train_exit_code -eq 0 ]; then
                # 2. 提取保存路径并预测
                CKPT_PATH=$(grep "Saved model to " "../../$LOG_FILE" | tail -n 1 | awk '{print $4}')
                
                # 如果 grep 失败，尝试默认路径构造 (需要包含fold信息如果文件名有的话，或者依赖grep)
                # wandb_train.py 的 save_model 通常会包含 fold 吗？需要确认，但 grep 应该最稳
                
                echo "[$(date)] Training process finished for Fold $FOLD." >> "../../$LOG_FILE"
            else
                echo "[$(date)] Training Failed for Fold $FOLD with code $train_exit_code" >> "../../$LOG_FILE"
            fi
        )
        
        echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=QID Fold=$FOLD"
    done
}

# 并行运行三个数据集 (GPU分配: 0, 1, 2)
# 注意：数据集名称必须与 data_config.json 中的键完全匹配 (区分大小写)

# GPU 0: XES3G5M (注意全大写)
run_dataset_experiments "XES3G5M" 0 &

# GPU 1: DBE_KT22 (注意全大写)
run_dataset_experiments "DBE_KT22" 1 &

# GPU 2: nips_task34 (注意小写，data_config中key为nips_task34)
run_dataset_experiments "nips_task34" 2 &

# 等待所有后台任务完成
echo "All QID experiments launched in parallel. Waiting for completion..."
wait
echo "All QID experiments finished."
