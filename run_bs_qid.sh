#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting QID experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 构造相对保存路径
    # 我们希望结果分别保存在 saved_model/bs/qid 目录下
    REL_SAVE_DIR="saved_model/bs/qid"
    mkdir -p "$REL_SAVE_DIR"
    
    (
        # 遍历 LSTM 层数 (串行)
        for LAYERS in 1 2 3; do
            # 构造日志文件名
            LOG_FILE="saved_model/bs/logs/qid_${DATASET}_lstm${LAYERS}.log"
            
            echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=QID Layers=$LAYERS"
            echo "[$(date)] Starting Training..." > "$LOG_FILE"
            
            # 切换到脚本目录执行 (保持相对路径一致性)
            (cd scripts_training2testing/examples && \
             # 1. 训练
             # 注意：使用 wandb_thinkkt_train.py 以获得更好的参数支持
             # d_knowledge=200 复刻 CRKT
             # batch_size=64 (QID模式显存占用小，可以使用默认64)
             python wandb_thinkkt_train.py \
                --dataset_name "$DATASET" \
                --question_rep_type "qid" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "../../$REL_SAVE_DIR" \
                --d_question 200 \
                --d_knowledge 200 \
                --gpu_id "$GPU_ID" \
                --use_cot 0 \
                --use_visual 0 \
                --num_epochs 200 \
                --batch_size 64 \
                --use_wandb 1 \
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
                        --use_wandb 1 \
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
            
            echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=QID Layers=$LAYERS"
        done
        
        echo "[GPU $GPU_ID] All layers completed for $DATASET"
    )
}

# 并行运行三个数据集 (GPU分配: 0, 1, 2)

# GPU 0: XES3G5M
run_dataset_experiments "xes3g5m" 0 &

# GPU 1: DBE_KT22
run_dataset_experiments "dbe_kt22" 1 &

# GPU 2: Eedi (NIPS_task34)
# 注意：Eedi 数据集在某些代码中可能被称为 nips_task34，请根据实际情况确认
# 根据 run_bs_qid.sh 原文使用的是 'eedi'，但 run_bs_experiments.sh 使用 'nips_task34'
# 我们这里保持与原脚本一致 'eedi'，但若有问题请尝试 'nips_task34'
run_dataset_experiments "eedi" 2 &

# 等待所有后台任务完成
echo "All QID experiments launched in parallel. Waiting for completion..."
wait
echo "All QID experiments finished."
