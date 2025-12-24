#!/bin/bash

# 创建日志目录
mkdir -p bs_logs

# 定义运行函数的逻辑
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting experiments for dataset: $DATASET on GPU: $GPU_ID"
    
    # 切换不同实验类型
    for Q_TYPE in "qid" "visual" "v&q"; do
        
        # 确定保存目录
        if [ "$Q_TYPE" == "qid" ]; then
            SAVE_DIR="bs_qid"
            # QID模式需要确保 d_question 正确 (这里脚本传参200以防万一，虽然代码内部已处理)
            # 但 wandb_thinkkt_train.py 默认 1024。
            # 我们的代码修改会 override config，所以这里传什么不重要，但为了清晰可以传 200
            D_QUESTION_ARG=200
        elif [ "$Q_TYPE" == "visual" ]; then
            SAVE_DIR="bs_visual"
            D_QUESTION_ARG=1024 # Visual features are 1024
        elif [ "$Q_TYPE" == "v&q" ]; then
            SAVE_DIR="bs_v&q" # 允许 & 符号
            D_QUESTION_ARG=1024 # Logic handles this
        fi
        
        # 确保保存目录存在 (相对于 working dir)
        mkdir -p "scripts_training2testing/examples/$SAVE_DIR"

        # 遍历 LSTM 层数
        for LAYERS in 1 2 3; do
            
            # 构造日志文件名 (处理 & 符号)
            SAFE_TYPE=$(echo $Q_TYPE | sed 's/&/_and_/g')
            LOG_FILE="bs_logs/${DATASET}_${SAFE_TYPE}_lstm${LAYERS}.log"
            
            echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=$Q_TYPE Layers=$LAYERS"
            echo "Logs: $LOG_FILE"
            
            # 运行命令
            # 注意: 需要在 scripts_training2testing/examples 目录下运行
            (cd scripts_training2testing/examples && \
             python wandb_thinkkt_train.py \
                --dataset_name "$DATASET" \
                --question_rep_type "$Q_TYPE" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "$SAVE_DIR" \
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
