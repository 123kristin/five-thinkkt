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
        
        # --- 断点续传检查 ---
        # 1. 尝试找到可能已存在的 checkpoint 目录 (假设命名规则)
        # 我们可以模糊匹配该 saved_model/bs/qid 下包含 "_qid_" 和此 dataset/fold 的目录
        # 或者更简单：如果日志文件里已经显示 "Prediction finished"，则跳过？
        # 用户要求的标准：Ckpt目录存在 且 包含 .ckpt 和 predicting*.log
        
        # 由于目录名包含动态参数，很难直接预测。
        # 策略：检查日志文件是否包含 "Prediction finished"。
        # 或者：遍历该目录下所有子目录，看是否有满足条件的。
        
        ALREADY_DONE=0
        # 查找该目录下所有子目录
        for DIR in "../../$REL_SAVE_DIR"/*/; do
            if [ -d "$DIR" ]; then
                #看目录名是否包含 dataset 和 fold (简单检查)
                if [[ "$DIR" == *"${DATASET}_${FOLD}_"* ]]; then
                     # 检查 .ckpt
                     if ls "$DIR"/*.ckpt 1> /dev/null 2>&1; then
                         # 检查 predicting*.log
                         if ls "$DIR"/predicting*.log 1> /dev/null 2>&1; then
                             ALREADY_DONE=1
                             break
                         fi
                     fi
                fi
            fi
        done
        
        if [ $ALREADY_DONE -eq 1 ]; then
            echo "[$(date)] Fold $FOLD already finished (Found .ckpt and predicting log). Skipping..." | tee -a "$LOG_FILE"
            continue
        fi
        # ------------------
        
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
                # 2. 提取保存路径
                # 必须精确匹配当前 Dataset 和 Fold，防止并发或乱序导致的目录混淆
                # 模式：*${DATASET}*${FOLD}* (假设params_str包含这两个)
                # wandb_train.py 中 params_str 是 params 中未被排除的 key 的 join
                # dataset_name 和 fold 都在 params 中且未被排除
                
                # 使用 grep 进一步过滤，确保 fold 匹配正确 (例如防止 fold=1 匹配到 fold=10，虽然这里只有0-4)
                # 注意：params_str 通常形如 {dataset}_{fold}_... 或 {model}_{dataset}_{fold}...
                
                CKPT_DIR=$(ls -td "../../$REL_SAVE_DIR"/*"${DATASET}"*"${FOLD}"* | head -1)
                
                if [ -n "$CKPT_DIR" ]; then
                    echo "Found checkpoint dir: $CKPT_DIR"
                    python wandb_predict.py \
                    --save_dir "$CKPT_DIR" \
                    --question_rep_type "qid" \
                    --dim_qc 200 \
                    --gpu_id "$GPU_ID" \
                    --use_wandb 0
                    
                    echo "[$(date)] Prediction finished for Fold $FOLD." >> "../../$LOG_FILE"
                else
                    echo "[$(date)] Could not find checkpoint dir in ../../$REL_SAVE_DIR" >> "../../$LOG_FILE"
                fi
                
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
