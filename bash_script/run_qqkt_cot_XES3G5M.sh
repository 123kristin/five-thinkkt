#!/bin/bash

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "Project Root: $PROJECT_ROOT"

# 定义绝对路径
LOG_DIR="$PROJECT_ROOT/saved_model/qqkt/logs"
SAVE_DIR="$PROJECT_ROOT/saved_model/qqkt/cot"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

DATASET="XES3G5M"
GPU_ID=3

echo "Starting QQKT CoT experiments for Dataset: $DATASET on GPU: $GPU_ID"

# 暂时只跑第 0 折
for FOLD in 0; do
    LOG_FILE="$LOG_DIR/cot_${DATASET}_fold${FOLD}.log"
    
    echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=CoT Fold=$FOLD"
    
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
        CURRENT_GPU_ID=$GPU_ID python wandb_qqkt_train.py \
        --dataset_name "$DATASET" \
        --model_name "qqkt" \
        --use_cot 1 \
        --fold "$FOLD" \
        --save_dir "$SAVE_DIR" \
        --dim_qc 200 \
        --d_cot 384 \
        --gpu_id "$GPU_ID" \
        --resume \
        --use_wandb 0 \
            >> "$LOG_FILE" 2>&1
        
        train_exit_code=$?
        
        if [ $train_exit_code -eq 0 ]; then
             # Predict
            CKPT_DIR=$(ls -td "$SAVE_DIR"/*"${DATASET}"*"${FOLD}"* | head -1)
            
            if [ -n "$CKPT_DIR" ]; then
                echo "Found checkpoint dir: $CKPT_DIR"
                CURRENT_GPU_ID=$GPU_ID python wandb_predict.py \
                --save_dir "$CKPT_DIR" \
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
    
    echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=CoT Fold=$FOLD"
done
