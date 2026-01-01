#!/bin/bash

# 获取项目根目录 (假设脚本在 bash_script/ 下，根目录在上级)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "Project Root: $PROJECT_ROOT"

# 定义绝对路径
LOG_DIR="$PROJECT_ROOT/saved_model/qqkt_cot/logs"
SAVE_DIR="$PROJECT_ROOT/saved_model/qqkt_cot/checkpoints"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# 定义运行实验的函数
run_experiment() {
    local DATASET=$1
    local GPU_ID=$2
    local FOLD=0  # 这里暂定只跑第0折，如果要跑所有折可以改成循环或者参数传入

    # 构造日志文件名 (绝对路径)
    LOG_FILE="$LOG_DIR/qqkt_cot_${DATASET}_fold${FOLD}.log"
    
    echo "[GPU $GPU_ID] Starting Training: Dataset=$DATASET Fold=$FOLD"
    echo "Log: $LOG_FILE"
    
    (
        # 切换到执行目录
        cd "$PROJECT_ROOT/scripts_training2testing/examples" && \
        
        # 启动训练
        # 注意：使用 nohup 调用时，显式传递 GPU ID 环境变量
        CURRENT_GPU_ID=$GPU_ID python wandb_qqkt_train.py \
            --dataset_name "$DATASET" \
            --use_cot 1 \
            --fold "$FOLD" \
            --save_dir "$SAVE_DIR" \
            --resume \
            --gpu_id "$GPU_ID" \
            --use_wandb 0 \
            >> "$LOG_FILE" 2>&1
            
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[$(date)] Training finished successfully for $DATASET Fold $FOLD" >> "$LOG_FILE"
        else
            echo "[$(date)] Training FAILED for $DATASET Fold $FOLD (Exit code: $exit_code)" >> "$LOG_FILE"
        fi
    )
}

# --- 并行任务调度 ---

echo "Launching experiments..."

# GPU 1: DBE_KT22 和 nips_task34 并行 (后台运行)
# 注意：这里我们在同一个GPU上起两个进程，可能会争抢资源，请确保显存足够
run_experiment "DBE_KT22" 1 &
PID1=$!
echo "Launched DBE_KT22 on GPU 1 (PID: $PID1)"

run_experiment "nips_task34" 1 &
PID2=$!
echo "Launched nips_task34 on GPU 1 (PID: $PID2)"

# GPU 3: XES3G5M (后台运行)
run_experiment "XES3G5M" 3 &
PID3=$!
echo "Launched XES3G5M on GPU 3 (PID: $PID3)"

echo "All training tasks launched in background."
