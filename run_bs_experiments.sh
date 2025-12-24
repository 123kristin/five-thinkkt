#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行函数的逻辑
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting experiments for dataset: $DATASET on GPU: $GPU_ID"
    
    # 切换不同实验类型
    for Q_TYPE in "qid" "visual" "v&q"; do
        
        # 确定保存目录名 (只是名字，不是全路径)
        if [ "$Q_TYPE" == "qid" ]; then
            TYPE_DIR="qid"
            D_QUESTION_ARG=200
        elif [ "$Q_TYPE" == "visual" ]; then
            TYPE_DIR="visual"
            D_QUESTION_ARG=200 # Use 200 to trigger adapter (1024->200)
        elif [ "$Q_TYPE" == "v&q" ]; then
            TYPE_DIR="v&q" # 允许 & 符号
            D_QUESTION_ARG=200 # Use 200 to trigger adapter (1024->200)
        fi
        
        # 确保保存目录存在 (相对于 root)
        # 目标: saved_model/bs/{qid, visual, v&q}
        REL_SAVE_DIR="saved_model/bs/$TYPE_DIR"
        mkdir -p "$REL_SAVE_DIR"

        # 遍历 LSTM 层数
        for LAYERS in 1 2 3; do
            
            # 构造日志文件名 (处理 & 符号)
            SAFE_TYPE=$(echo $Q_TYPE | sed 's/&/_and_/g')
            LOG_FILE="saved_model/bs/logs/${DATASET}_${SAFE_TYPE}_lstm${LAYERS}.log"
            
            echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=$Q_TYPE Layers=$LAYERS"
            echo "Logs: $LOG_FILE"
            echo "Save Dir: $REL_SAVE_DIR"
            
            # 运行命令
            # 注意: 需要在 scripts_training2testing/examples 目录下运行
            # save_dir 需要传递相对于 examples 的路径，即 ../../saved_model/bs/xxx
            (cd scripts_training2testing/examples && \
             python wandb_thinkkt_train.py \
                --dataset_name "$DATASET" \
                --question_rep_type "$Q_TYPE" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "../../$REL_SAVE_DIR" \
                --d_question $D_QUESTION_ARG \
                --gpu_id "$GPU_ID" \
                --use_cot 0 \
                --use_visual 1 \
                --num_epochs 200 \
                 > "../../$LOG_FILE" 2>&1)
            
            echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=$Q_TYPE Layers=$LAYERS"
        done
    done
}

# 并行运行三个数据集
# Dataset 1 -> GPU 0
run_dataset_experiments "DBE_KT22" 0 &

# Dataset 2 -> GPU 1
run_dataset_experiments "XES3G5M" 1 &

# Dataset 3 -> GPU 2
run_dataset_experiments "nips_task34" 2 &

# 等待所有后台任务完成
wait

echo "All BS experiments completed."
