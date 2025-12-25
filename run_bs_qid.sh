#!/bin/bash

# Datasets and their corresponding GPU assignments
# Format: "dataset_name:gpu_id"
datasets=("xes3g5m:0" "dbe_kt22:1" "eedi:2")

# LSTM Layers to test indicating model depth
layers=(1 2 3)

# Base arguments
# CRKT Baseline: d_knowledge=200
ARGS="--model_name thinkkt \
      --emb_type qkcs \
      --save_dir saved_model \
      --learning_rate 1e-3 \
      --num_epochs 200 \
      --question_rep_type qid \
      --use_visual 0 \
      --d_knowledge 200 \
      --use_wandb 1"

echo "Starting QID Baseline Experiments..."

# Iterate over datasets (Outer loop - parallelism across GPUs)
for data_pair in "${datasets[@]}"; do
    # We spawn a background subshell for each GPU
    (
        IFS=':' read -r dataset gpu_id <<< "$data_pair"
        echo "Launching experiments for dataset: $dataset on GPU: $gpu_id"
        
        # Iterate over layers (Inner loop - Sequential WITHIN GPU to avoid OOM)
        for layer in "${layers[@]}"; do
            echo "  [GPU $gpu_id] Running $dataset - Layer $layer..."
            
            # Unique log file for this run
            LOG_FILE="saved_model/bs/logs/qid_${dataset}_layer${layer}.log"
            mkdir -p saved_model/bs/logs
            
            echo "[$(date)] Starting Training: ${dataset} Layer ${layer}" >> "$LOG_FILE"
            
            # 1. Training
            python -u scripts_training2testing/examples/wandb_train.py \
                $ARGS \
                --dataset_name $dataset \
                --num_lstm_layers $layer \
                --fold 0 \
                --gpu_id $gpu_id \
                >> "$LOG_FILE" 2>&1
            
            train_exit_code=$?
            
            if [ $train_exit_code -eq 0 ]; then
                echo "[$(date)] Training Finished. Starting Prediction..." >> "$LOG_FILE"
                
                # 2. Prediction
                # Construct save path dynamically to match training output
                # Typical pattern: save_dir + / + model_name + _ + dataset_name + ...
                # Assuming standard output struct, passing base dir is safer if predict finds latest
                
                # Check for the correct model save path logic or pass specific path if known
                # Using wildcards or standard naming convention
                
                python -u scripts_training2testing/examples/wandb_predict.py \
                    --save_dir "saved_model" \
                    --gpu_id $gpu_id \
                    >> "$LOG_FILE" 2>&1
                    
                echo "[$(date)] Prediction Finished." >> "$LOG_FILE"
            else
                echo "[$(date)] Training Failed with code $train_exit_code" >> "$LOG_FILE"
            fi
            
            echo "  [GPU $gpu_id] Finished $dataset - Layer $layer"
        done
    ) & 
done

# Wait for all 3 GPU jobs to finish
echo "All experiments launched in parallel on 3 GPUs. Waiting for completion..."
wait
echo "All QID experiments finished."
