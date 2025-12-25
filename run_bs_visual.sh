#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting Visual experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 构造相对保存路径
    # 我们希望结果分别保存在 saved_model/bs/visual 目录下
    REL_SAVE_DIR="saved_model/bs/visual"
    mkdir -p "$REL_SAVE_DIR"
    
    (
        # 遍历 LSTM 层数 (串行)
        for LAYERS in 1 2 3; do
            # 构造日志文件名
            LOG_FILE="saved_model/bs/logs/visual_${DATASET}_lstm${LAYERS}.log"
            
            echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=Visual Layers=$LAYERS"
            echo "[$(date)] Starting Training..." > "$LOG_FILE"
            
            # 切换到脚本目录执行 (保持相对路径一致性)
            (cd scripts_training2testing/examples && \
             # 1. 训练
             # d_knowledge=200 复刻 CRKT
             # use_visual=1, question_rep_type=visual
             # 不指定 batch_size，让代码由 data_config/train_config 决定 (Visual通常限制为32以防OOM)
             python wandb_thinkkt_train.py \
                --dataset_name "$DATASET" \
                --question_rep_type "visual" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "../../$REL_SAVE_DIR" \
                --d_question 200 \
                --d_knowledge 200 \
                --gpu_id "$GPU_ID" \
                --use_cot 0 \
                --use_visual 1 \
                --num_epochs 200 \
                --use_wandb 0 \
                 >> "../../$LOG_FILE" 2>&1
             
             train_exit_code=$?
             
             if [ $train_exit_code -eq 0 ]; then
                 # 2. 提取保存路径并预测
                 CKPT_PATH=$(grep "模型目录: " "../../$LOG_FILE" | tail -n 1 | awk '{print $2}')
                 
                 if [ ! -z "$CKPT_PATH" ]; then
                     echo "[$(date)] Training Finished. Found Checkpoint: $CKPT_PATH" >> "../../$LOG_FILE"
                     echo "Starting Prediction..." >> "../../$LOG_FILE"
                     
                     python wandb_predict.py \
                        --save_dir "$CKPT_PATH" \
                        --gpu_id "$GPU_ID" \
                        --use_wandb 0 \
                        --bz 128 \
                        >> "../../$LOG_FILE" 2>&1
                        
                     echo "[$(date)] Prediction Finished." >> "../../$LOG_FILE"
                 else
                     echo "Error: Could not find Checkpoint Path in log file!" >> "../../$LOG_FILE"
                 fi
             else
                 echo "[$(date)] Training Failed with code $train_exit_code" >> "../../$LOG_FILE"
             fi
            )
            
            echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=Visual Layers=$LAYERS"
        done
        
        echo "[GPU $GPU_ID] All layers completed for $DATASET"
    )
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
echo "All Visual experiments launched in parallel. Waiting for completion..."
wait
echo "All Visual experiments finished."
