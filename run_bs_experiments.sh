#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行函数的逻辑
run_type_experiments() {
    local Q_TYPE=$1
    local GPU_ID=$2
    
    echo "Starting experiments for Type: $Q_TYPE on GPU: $GPU_ID"
    
    # 确定此类型的配置
    if [ "$Q_TYPE" == "qid" ]; then
        TYPE_DIR="qid"
        D_QUESTION_ARG=200
    elif [ "$Q_TYPE" == "visual" ]; then
        TYPE_DIR="visual"
        D_QUESTION_ARG=200
    elif [ "$Q_TYPE" == "v&q" ]; then
        TYPE_DIR="v&q"
        D_QUESTION_ARG=200
    fi
    
    # 确保保存目录存在
    REL_SAVE_DIR="saved_model/bs/$TYPE_DIR"
    mkdir -p "$REL_SAVE_DIR"

    # 遍历数据集 - 并行运行
    for DATASET in "DBE_KT22" "XES3G5M" "nips_task34"; do
        (
            echo "Starting batch for $DATASET - $Q_TYPE on GPU $GPU_ID"
            
            # 遍历 LSTM 层数 (串行)
            for LAYERS in 1 2 3; do
                
                # 构造日志文件名
                SAFE_TYPE=$(echo $Q_TYPE | sed 's/&/_and_/g')
                LOG_FILE="saved_model/bs/logs/${DATASET}_${SAFE_TYPE}_lstm${LAYERS}.log"
                
                echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=$Q_TYPE Layers=$LAYERS"
                
                # 运行命令
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
        ) & # Put dataset loop in background
    done
    
    # Wait for all datasets for this Type on this GPU
    wait
}

# 并行运行三种模式
# GPU 0: QID (All Datasets)
run_type_experiments "qid" 0 &

# GPU 1: Visual (All Datasets)
run_type_experiments "visual" 1 &

# GPU 2: V&Q (All Datasets)
run_type_experiments "v&q" 2 &

# 等待所有后台任务完成
wait

echo "All BS experiments completed."
