#!/bin/bash

# 创建日志目录
mkdir -p saved_model/cot/logs

# 定义运行函数
run_experiment_on_gpu() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting tasks for $DATASET on GPU $GPU_ID"
    
    # 这里的 D_QUESTION_ARG 在 QID 模式下是 200
    # 在 Visual 模式下，如果是 Visual Only 是 200 (projected), V&Q 是 400
    # 但是 wandb_thinkkt_train.py 接收 d_question 参数
    # run_bs 里面写的是: QID=200, Visual=200, V&Q=200.
    # 这意味着 ThinkKT 内部自动处理维度。如果是 V&Q，内部 net_d_question 会变成 400。
    # 所以我们这里传 200 即可。
    D_QUESTION_ARG=200
    
    # 定义 CoT 缓存目录 (每个数据集独立，避免冲突)
    COT_CACHE_DIR="cot_cache/cot_cache_${DATASET}"
    mkdir -p "$COT_CACHE_DIR"
    
    # ---------------------------------------------------------
    # 第一阶段：生成 CoT 缓存 (Base Case: QID, Layer 1)
    # ---------------------------------------------------------
    echo "[GPU $GPU_ID] Phase 1: Generating CoT Cache using QID + LSTM Layer 1"
    
    Q_TYPE="qid"
    LAYERS=1
    
    # 1. Check Completion
    REL_SAVE_DIR="saved_model/cot/${Q_TYPE}"
    mkdir -p "$REL_SAVE_DIR"
    LOG_FILE="saved_model/cot/logs/${DATASET}_${Q_TYPE}_lstm${LAYERS}.log"
    
    # Check if completed
    (cd scripts_training2testing/examples && \
     python check_completion.py \
        --dataset_name "$DATASET" \
        --question_rep_type "$Q_TYPE" \
        --num_lstm_layers "$LAYERS" \
        --save_dir "../../$REL_SAVE_DIR" \
        --d_question $D_QUESTION_ARG \
        --use_cot 1 \
        --use_visual 1 \
        --num_epochs 200)
    
    if [ $? -eq 0 ]; then
        echo "[GPU $GPU_ID] Phase 1 Already Completed. Cache should exist."
    else
        # Run Training
        echo "[GPU $GPU_ID] Running Phase 1: $DATASET $Q_TYPE L$LAYERS"
        (cd scripts_training2testing/examples && \
         # Training
         echo "Starting Training..." > "../../$LOG_FILE"
         python wandb_thinkkt_train.py \
            --dataset_name "$DATASET" \
            --question_rep_type "$Q_TYPE" \
            --num_lstm_layers "$LAYERS" \
            --save_dir "../../$REL_SAVE_DIR" \
            --d_question $D_QUESTION_ARG \
            --gpu_id "$GPU_ID" \
            --use_cot 1 \
            --cot_cache_dir "../../$COT_CACHE_DIR" \
            --cot_threshold 2 \
            --adaptive_strategy rule \
            --use_visual 1 \
            --num_epochs 200 \
             >> "../../$LOG_FILE" 2>&1
         
         # Prediction
         CKPT_PATH=$(grep "模型目录: " "../../$LOG_FILE" | tail -n 1 | awk '{print $2}')
         if [ ! -z "$CKPT_PATH" ]; then
             echo "Training Finished. Found Checkpoint: $CKPT_PATH" >> "../../$LOG_FILE"
             python wandb_predict.py --save_dir "$CKPT_PATH" --gpu_id "$GPU_ID" --bz 4 --use_wandb 0 >> "../../$LOG_FILE" 2>&1
         fi
        )
    fi
    
    # ---------------------------------------------------------
    # 第二阶段：运行其他所有组合 (Visual, V&Q, Layers 2,3)
    # ---------------------------------------------------------
    echo "[GPU $GPU_ID] Phase 2: Running other configurations..."
    
    for Q_TYPE in "qid" "visual" "v&q"; do
        for LAYERS in 1 2 3; do
            # Skip Phase 1 Case (QID + Layer 1)
            if [ "$Q_TYPE" == "qid" ] && [ "$LAYERS" == "1" ]; then
                continue
            fi
            
            SAFE_TYPE=$(echo $Q_TYPE | sed 's/&/_and_/g') # for filename
            REL_SAVE_DIR="saved_model/cot/${Q_TYPE}"
            mkdir -p "$REL_SAVE_DIR"
            LOG_FILE="saved_model/cot/logs/${DATASET}_${SAFE_TYPE}_lstm${LAYERS}.log"

            # Check Completion
            (cd scripts_training2testing/examples && \
             python check_completion.py \
                --dataset_name "$DATASET" \
                --question_rep_type "$Q_TYPE" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "../../$REL_SAVE_DIR" \
                --d_question $D_QUESTION_ARG \
                --use_cot 1 \
                --use_visual 1 \
                --num_epochs 200)

            if [ $? -eq 0 ]; then
                echo "[GPU $GPU_ID] Skipping completed: $DATASET $Q_TYPE L$LAYERS"
                continue
            fi
            
            echo "[GPU $GPU_ID] Running: $DATASET $Q_TYPE L$LAYERS"
            (cd scripts_training2testing/examples && \
             echo "Starting Training..." > "../../$LOG_FILE"
             python wandb_thinkkt_train.py \
                --dataset_name "$DATASET" \
                --question_rep_type "$Q_TYPE" \
                --num_lstm_layers "$LAYERS" \
                --save_dir "../../$REL_SAVE_DIR" \
                --d_question $D_QUESTION_ARG \
                --gpu_id "$GPU_ID" \
                --use_cot 1 \
                --cot_cache_dir "../../$COT_CACHE_DIR" \
                --cot_threshold 2 \
                --adaptive_strategy rule \
                --use_visual 1 \
                --num_epochs 200 \
                 >> "../../$LOG_FILE" 2>&1
             
             CKPT_PATH=$(grep "模型目录: " "../../$LOG_FILE" | tail -n 1 | awk '{print $2}')
             if [ ! -z "$CKPT_PATH" ]; then
                 echo "Training Finished. Checkpoint: $CKPT_PATH" >> "../../$LOG_FILE"
                 python wandb_predict.py --save_dir "$CKPT_PATH" --gpu_id "$GPU_ID" --bz 4 --use_wandb 0 >> "../../$LOG_FILE" 2>&1
             fi
            )
        done
    done
    
    echo "[GPU $GPU_ID] All experiments for $DATASET finished."
}

# ---------------------------------------------------------
# 并行启动任务
# ---------------------------------------------------------

# GPU 0: DBE_KT22 (注意：DBE 数据集名称在wandb_train中会被映射，这里传 DBE_KT22)
(run_experiment_on_gpu "DBE_KT22" 0) &

# GPU 1: XES3G5M
(run_experiment_on_gpu "XES3G5M" 1) &

# GPU 2: nips_task34
(run_experiment_on_gpu "nips_task34" 2) &

# 等待所有后台任务结束
wait
echo "All experiments completed."
