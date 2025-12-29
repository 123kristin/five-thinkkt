#!/bin/bash

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "Project Root: $PROJECT_ROOT"

# 定义绝对路径
LOG_DIR="$PROJECT_ROOT/saved_model/bs/logs"
SAVE_DIR="$PROJECT_ROOT/saved_model/bs/cl"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# 定义运行实验的函数 (训练完立即测试)
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting VCRKT CL experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 遍历 5 折 (串行)
    for FOLD in 0 1 2 3 4; do
        LOG_FILE="$LOG_DIR/cl_${DATASET}_fold${FOLD}.log"
        
        echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=CL Fold=$FOLD"
        
        # --- 断点续传检查 ---
        ALREADY_DONE=0
        for DIR in "$SAVE_DIR"/*/; do
            if [ -d "$DIR" ]; then
                if [[ "$DIR" == *"${DATASET}_${FOLD}_"* ]]; then
                     if ls "$DIR"/*.ckpt 1> /dev/null 2>&1; then
                         if ls "$DIR"/predicting*.log 1> /dev/null 2>&1; then
                             ALREADY_DONE=1
                             break
                         fi
                     fi
                fi
            fi
        done
        
        if [ $ALREADY_DONE -eq 1 ]; then
            echo "[$(date)] Fold $FOLD already finished. Skipping..." | tee -a "$LOG_FILE"
            continue
        fi
        # ------------------
        
        echo "[$(date)] Starting Training Fold $FOLD..." > "$LOG_FILE"
        
        (
            cd "$PROJECT_ROOT/scripts_training2testing/examples" && \
            # 1. 训练
            python wandb_vcrkt_train.py \
            --dataset_name "$DATASET" \
            --model_name "vcrkt" \
            --question_rep_type "cl" \
            --cl_weight 0.1 \
            --fold "$FOLD" \
            --save_dir "$SAVE_DIR" \
            --dim_qc 200 \
            --d_question 1024 \
            --gpu_id "$GPU_ID" \
            --num_epochs 200 \
            --use_wandb 0 \
                >> "$LOG_FILE" 2>&1
            
            train_exit_code=$?
            
            if [ $train_exit_code -eq 0 ]; then
                 # 2. Predict (训练成功后立即预测)
                CKPT_DIR=$(ls -td "$SAVE_DIR"/*"${DATASET}"*"${FOLD}"* | head -1)
                
                if [ -n "$CKPT_DIR" ]; then
                    echo "Found checkpoint dir: $CKPT_DIR"
                    python wandb_predict.py \
                    --save_dir "$CKPT_DIR" \
                    --question_rep_type "cl" \
                    --cl_weight 0.1 \
                    --d_question 1024 \
                    --dim_qc 200 \
                    --gpu_id "$GPU_ID" \
                    --bz 128 \
                    --use_wandb 0
                    
                    echo "[$(date)] Prediction finished for Fold $FOLD." >> "$LOG_FILE"
                else
                    echo "[$(date)] Could not find checkpoint dir" >> "$LOG_FILE"
                fi
                
                echo "[$(date)] Training process for Fold $FOLD finished." >> "$LOG_FILE"
            else
                echo "[$(date)] Training Failed for Fold $FOLD with code $train_exit_code" >> "$LOG_FILE"
            fi
        )
        
        echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=CL Fold=$FOLD"
    done
}

# 并行运行逻辑
# 任务1: XES3G5M 在 GPU 0 上独立运行
run_dataset_experiments "XES3G5M" 0 &

# 任务2: DBE_KT22 和 nips_task34 在 GPU 1 上串行运行
(
    run_dataset_experiments "DBE_KT22" 1
    run_dataset_experiments "nips_task34" 1
) &

# 等待所有后台任务完成
echo "All CL experiments launched. XES on GPU 0, DBE+NIPS on GPU 1. Waiting..."
wait
echo "All CL experiments finished."
