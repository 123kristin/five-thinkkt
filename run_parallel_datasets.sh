#!/bin/bash
echo "Starting Parallel Experiments by Dataset"
mkdir -p logs

# ================= 配置区域 (Configuration) =================
# 是否使用思维链 (Chain of Thought)
# 可选: 0 (不使用/Baseline) 或 1 (使用CoT)
USE_COT=0
# USE_COT=1

# 适应性策略 (仅当 USE_COT=1 时有效)
ADAPTIVE_STRATEGY="rule"
# ============================================================

echo "Configuration:"
echo "  Use CoT:      ${USE_COT}"
if [ "${USE_COT}" -eq 1 ]; then
    echo "  Strategy:     ${ADAPTIVE_STRATEGY}"
fi
echo "Running Experiments: CRKT Baseline (QID) + Visual Baseline (Visual) x LSTM Layers [1,2,3]"
echo "------------------------------------------------"

# --- GPU 0: DBE_KT22 ---
echo "Launching DBE_KT22 (CRKT + Visual) on GPU 0..."
# 1. CRKT Baseline (QID) - Exp 1,2,3
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 0 \
    --experiment_range "1,2,3" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/dbe_crkt.log 2>&1 &

# 2. Visual Baseline (Visual) - Exp 4,5,6
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 0 \
    --experiment_range "4,5,6" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/dbe_visual.log 2>&1 &


# --- GPU 1: XES3G5M ---
echo "Launching XES3G5M (CRKT + Visual) on GPU 1..."
# 1. CRKT Baseline (QID) - Exp 7,8,9
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 1 \
    --experiment_range "7,8,9" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/xes_crkt.log 2>&1 &

# 2. Visual Baseline (Visual) - Exp 10,11,12
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 1 \
    --experiment_range "10,11,12" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/xes_visual.log 2>&1 &


# --- GPU 2: nips_task34 ---
echo "Launching nips_task34 (CRKT + Visual) on GPU 2..."
# 1. CRKT Baseline (QID) - Exp 13,14,15
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 2 \
    --experiment_range "13,14,15" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/nips_crkt.log 2>&1 &

# 2. Visual Baseline (Visual) - Exp 16,17,18
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 2 \
    --experiment_range "16,17,18" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > logs/nips_visual.log 2>&1 &


echo "All jobs launched in background."
echo "Check logs/dbe_exp.log, logs/xes_exp.log, logs/nips_exp.log for progress."
echo "Use 'nvidia-smi' to monitor GPU usage."
