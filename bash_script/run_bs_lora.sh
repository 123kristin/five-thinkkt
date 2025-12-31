#!/bin/bash

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "Project Root: $PROJECT_ROOT"

# 定义绝对路径
LOG_DIR="$PROJECT_ROOT/saved_model/bs/logs"
SAVE_DIR="$PROJECT_ROOT/saved_model/bs/lora"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# 定义运行实验的函数 (训练完立即测试)
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting VCRKT QLoRA experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 遍历 5 折 (串行)
    for FOLD in 0 1 2 3 4; do
        LOG_FILE="$LOG_DIR/lora_${DATASET}_fold${FOLD}.log"
        
        echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=GF Mode=QLoRA Fold=$FOLD"
        
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
        
        echo "[$(date)] Starting QLoRA Training Fold $FOLD..." > "$LOG_FILE"
        
        # 注意: LoRA 训练显存开销较大，且必须在线跑大模型，速度较慢
        (
            cd "$PROJECT_ROOT/scripts_training2testing/examples" && \
            python wandb_vcrkt_train.py \
            --dataset_name "$DATASET" \
            --model_name "vcrkt" \
            --question_rep_type "gf" \
            --use_lora 1 \
            --lora_r 16 \
            --fold "$FOLD" \
            --save_dir "$SAVE_DIR" \
            --dim_qc 200 \
            --d_question 1024 \
            --gpu_id "$GPU_ID" \
            --num_epochs 100 \
            --use_wandb 0 \
                >> "$LOG_FILE" 2>&1
            
            train_exit_code=$?
            
            if [ $train_exit_code -eq 0 ]; then
                 # 2. Predict (训练成功后立即预测)
                CKPT_DIR=$(ls -td "$SAVE_DIR"/*"${DATASET}"*"${FOLD}"* | head -1)
                
                if [ -n "$CKPT_DIR" ]; then
                    echo "Found checkpoint dir: $CKPT_DIR"
                    
                    # 预测时也需要开启 use_lora 以加载适配器
                    # 注意: wandb_predict.py 也需要支持 use_lora 参数? 
                    # 暂时 train_one_step/predict_one_step 是共用逻辑，只要 model 加载对了就行
                    
                    python wandb_predict.py \
                    --save_dir "$CKPT_DIR" \
                    --question_rep_type "gf" \
                    --d_question 1024 \
                    --dim_qc 200 \
                    --gpu_id "$GPU_ID" \
                    --bz 32 \
                    --use_wandb 0 \
                    --use_lora 1
                    # 注意: 预测时batch size可以小一点，因为需要在线推理
                    
                    echo "[$(date)] Prediction finished for Fold $FOLD." >> "$LOG_FILE"
                else
                    echo "[$(date)] Could not find checkpoint dir" >> "$LOG_FILE"
                fi
                
                echo "[$(date)] Training process for Fold $FOLD finished." >> "$LOG_FILE"
            else
                echo "[$(date)] Training Failed for Fold $FOLD with code $train_exit_code" >> "$LOG_FILE"
            fi
        )
        
        echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=GF Mode=QLoRA Fold=$FOLD"
    done
}

# 并行运行三个数据集 (GPU分配: 0, 1, 2)
# 分配到不同GPU，因为大模型加载显存占用

# GPU 0: XES3G5M
run_dataset_experiments "XES3G5M" 1 &

# GPU 1: DBE_KT22
run_dataset_experiments "DBE_KT22" 2 &

# GPU 2: nips_task34
run_dataset_experiments "nips_task34" 3 &

# 等待所有后台任务完成
echo "All QLoRA experiments launched in parallel. Waiting for completion..."
wait
echo "All QLoRA experiments finished."
