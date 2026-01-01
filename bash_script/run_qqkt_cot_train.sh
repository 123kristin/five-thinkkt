#!/bin/bash

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "Project Root: $PROJECT_ROOT"

# 定义绝对路径 (参照 run_bs_cl.sh)
LOG_DIR="$PROJECT_ROOT/saved_model/qqkt/logs"
SAVE_DIR="$PROJECT_ROOT/saved_model/qqkt/cot"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting QQKT CoT experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 暂时只跑第 0 折, 如需跑所有折可改为: for FOLD in 0 1 2 3 4; do
    for FOLD in 0; do
        LOG_FILE="$LOG_DIR/cot_${DATASET}_fold${FOLD}.log"
        
        echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=CoT Fold=$FOLD"
        
        # --- 断点续传检查 (参照 run_bs_cl.sh) ---
        ALREADY_DONE=0
        for DIR in "$SAVE_DIR"/*/; do
            if [ -d "$DIR" ]; then
                # 检查是否存在对应数据集和折数的目录
                if [[ "$DIR" == *"${DATASET}_${FOLD}_"* ]]; then
                     # 检查是否有ckpt文件
                     if ls "$DIR"/*.ckpt 1> /dev/null 2>&1; then
                         # 检查是否有预测日志 (代表已完成预测)
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
            # 注意: 显式传递 GPU ID 并使用 --resume
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
                 # 2. Predict (训练成功后立即预测)
                 # 寻找该折最新的 checkpoint 目录
                CKPT_DIR=$(ls -td "$SAVE_DIR"/*"${DATASET}"*"${FOLD}"* | head -1)
                
                if [ -n "$CKPT_DIR" ]; then
                    echo "Found checkpoint dir: $CKPT_DIR"
                    # 注意: 预测脚本也需要指定 GPU
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
}

# 并行运行逻辑

echo "Launching experiments..."

# 任务1: XES3G5M 在 GPU 3 上独立运行 (后台)
run_dataset_experiments "XES3G5M" 3 &
PID3=$!
echo "Launched XES3G5M on GPU 3 (PID: $PID3)"

# 任务2: DBE_KT22 和 nips_task34 在 GPU 1 上串行运行 (整体放入后台)
# 这样可以保证两者依次使用 GPU 1，不会OOM，同时不阻塞当前终端
(
    run_dataset_experiments "DBE_KT22" 1
    run_dataset_experiments "nips_task34" 1
) &
PID1=$!
echo "Launched DBE_KT22 then nips_task34 on GPU 1 (PID: $PID1)"

# 等待提示
echo "All tasks launched in background."
