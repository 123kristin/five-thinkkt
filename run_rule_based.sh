#!/bin/bash
# Parallel run for Rule-Based on 4 GPUs
echo "Starting Rule-Based experiments on 4 GPUs..."

# GPU 0: Experiments 1-5
python scripts/run_all_thinkkt_experiments.py --gpu_id 0 --experiment_range "1,2,3,4,5" --use_cot 1 --adaptive_strategy rule > logs/rule_gpu0.log 2>&1 &
PID0=$!

# GPU 1: Experiments 6-10
python scripts/run_all_thinkkt_experiments.py --gpu_id 1 --experiment_range "6,7,8,9,10" --use_cot 1 --adaptive_strategy rule > logs/rule_gpu1.log 2>&1 &
PID1=$!

# GPU 2: Experiments 11-14
python scripts/run_all_thinkkt_experiments.py --gpu_id 2 --experiment_range "11,12,13,14" --use_cot 1 --adaptive_strategy rule > logs/rule_gpu2.log 2>&1 &
PID2=$!

# GPU 3: Experiments 15-18
python scripts/run_all_thinkkt_experiments.py --gpu_id 3 --experiment_range "15,16,17,18" --use_cot 1 --adaptive_strategy rule > logs/rule_gpu3.log 2>&1 &
PID3=$!

echo "Jobs launched: $PID0, $PID1, $PID2, $PID3"
wait
echo "All Rule-Based experiments completed."
