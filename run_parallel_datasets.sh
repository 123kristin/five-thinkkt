#!/bin/bash
echo "Starting Parallel Experiments by Dataset (Baseline + Rule-Based)"
mkdir -p logs

# --- GPU 0: DBE_KT22 (Experiments 1-6) ---
echo "Launching DBE_KT22 on GPU 0..."
# Baseline
python scripts/run_all_thinkkt_experiments.py --gpu_id 0 --experiment_range "1,2,3,4,5,6" --use_cot 0 > logs/dbe_baseline.log 2>&1 &
# Rule-Based
python scripts/run_all_thinkkt_experiments.py --gpu_id 0 --experiment_range "1,2,3,4,5,6" --use_cot 1 --adaptive_strategy rule > logs/dbe_rule.log 2>&1 &


# --- GPU 1: XES3G5M (Experiments 7-12) ---
echo "Launching XES3G5M on GPU 1..."
# Baseline
python scripts/run_all_thinkkt_experiments.py --gpu_id 1 --experiment_range "7,8,9,10,11,12" --use_cot 0 > logs/xes_baseline.log 2>&1 &
# Rule-Based
python scripts/run_all_thinkkt_experiments.py --gpu_id 1 --experiment_range "7,8,9,10,11,12" --use_cot 1 --adaptive_strategy rule > logs/xes_rule.log 2>&1 &


# --- GPU 2: nips_task34 (Experiments 13-18) ---
echo "Launching nips_task34 on GPU 2..."
# Baseline
python scripts/run_all_thinkkt_experiments.py --gpu_id 2 --experiment_range "13,14,15,16,17,18" --use_cot 0 > logs/nips_baseline.log 2>&1 &
# Rule-Based
python scripts/run_all_thinkkt_experiments.py --gpu_id 2 --experiment_range "13,14,15,16,17,18" --use_cot 1 --adaptive_strategy rule > logs/nips_rule.log 2>&1 &

echo "All jobs launched in background."
echo "Check logs/ folder for progress."
echo "Use 'nvidia-smi' to monitor GPU usage. If OOM occurs, kill jobs and run serially."
wait
