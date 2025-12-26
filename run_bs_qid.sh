#!/bin/bash

# 创建日志目录
mkdir -p saved_model/bs/logs

# 定义运行实验的函数
run_dataset_experiments() {
    local DATASET=$1
    local GPU_ID=$2
    
    echo "Starting VCRKT QID experiments for Dataset: $DATASET on GPU: $GPU_ID"
    
    # 构造相对保存路径
    REL_SAVE_DIR="saved_model/bs/qid"
    mkdir -p "$REL_SAVE_DIR"
    
    # 构造日志文件名
    LOG_FILE="saved_model/bs/logs/qid_${DATASET}.log"
    
    echo "[GPU $GPU_ID] Running: Dataset=$DATASET Type=QID"
    echo "[$(date)] Starting Training..." > "$LOG_FILE"
    
    (
        # 切换到脚本目录执行 (保持相对路径一致性)
        cd scripts_training2testing/examples && \
        # 1. 训练
        python wandb_vcrkt_train.py \
        --dataset_name "$DATASET" \
        --model_name "vcrkt" \
        --question_rep_type "qid" \
        --save_dir "../../$REL_SAVE_DIR" \
        --dim_qc 200 \
        --gpu_id "$GPU_ID" \
        --num_epochs 200 \
        --use_wandb 0 \
            >> "../../$LOG_FILE" 2>&1
        
        train_exit_code=$?
        
        if [ $train_exit_code -eq 0 ]; then
            # 2. 提取保存路径并预测
            CKPT_PATH=$(grep "Saved model to " "../../$LOG_FILE" | tail -n 1 | awk '{print $4}')
            # 这里的grep模式可能需要根据 wandb_train.py 的实际输出调整，暂时假设标准输出
            # 实际上 wandb_train.py 的 save_model 会打印路径，或者我们可以直接构建路径
            # 为了保险起见，我们假设 wandb_vcrkt_train.py 也会保存 best_model.ckpt
            
            # 如果 grep 失败，尝试默认路径构造
            if [ -z "$CKPT_PATH" ]; then
                 CKPT_PATH="../../$REL_SAVE_DIR/${DATASET}_0_0.001_64_vcrkt_qkcs_0.1_200_qid_1024"
                 # 注意：上面的路径名字构造可能不准确，依赖于 params_str
            fi

            # 由于 VCRKT 没有单独的 predict 脚本，通常 train 脚本结束时会跑测试
            # 如果需要单独跑 predict，可能需要修改 wandb_vcrkt_train.py 支持 --mode predict
            # 目前 VCRKT 的 wrapper 似乎包含了 predict_one_step，但脚本 wandb_train.py 主流程通常包含测试
            
            echo "[$(date)] Training process finished." >> "../../$LOG_FILE"
        else
            echo "[$(date)] Training Failed with code $train_exit_code" >> "../../$LOG_FILE"
        fi
    )
    
    echo "[GPU $GPU_ID] Finished: Dataset=$DATASET Type=QID"
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
echo "All QID experiments launched in parallel. Waiting for completion..."
wait
echo "All QID experiments finished."
