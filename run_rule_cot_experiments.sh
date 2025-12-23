#!/bin/bash
echo "Starting ThinkKT Rule-Based CoT Experiments"
mkdir -p cot_logs

# ================= 配置区域 =================
# 开启 CoT
USE_COT=1
# 使用规则策略
ADAPTIVE_STRATEGY="rule"
# ==========================================

echo "Configuration:"
echo "  Use CoT:      ${USE_COT}"
echo "  Strategy:     ${ADAPTIVE_STRATEGY}"
echo "Running Experiments: CRKT+RuleCoT AND Visual+RuleCoT (LSTM Layers 1,2,3)"
echo "------------------------------------------------"

# --- GPU 0: DBE_KT22 ---
echo "Launching DBE_KT22 on GPU 0..."
# 1. CRKT (QID) + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 1 \
    --experiment_range "1,2,3" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/dbe_crkt_rule_cot.log 2>&1 &

# 2. Visual + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 1 \
    --experiment_range "4,5,6" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/dbe_visual_rule_cot.log 2>&1 &


# --- GPU 1: XES3G5M ---
echo "Launching XES3G5M on GPU 1..."
# 1. CRKT (QID) + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 0 \
    --experiment_range "7,8,9" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/xes_crkt_rule_cot.log 2>&1 &

# 2. Visual + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 0 \
    --experiment_range "10,11,12" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/xes_visual_rule_cot.log 2>&1 &


# --- GPU 2: nips_task34 ---
echo "Launching nips_task34 on GPU 2..."
# 1. CRKT (QID) + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 2 \
    --experiment_range "13,14,15" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/nips_crkt_rule_cot.log 2>&1 &

# 2. Visual + Rule CoT
nohup python scripts/run_all_thinkkt_experiments.py \
    --gpu_id 2 \
    --experiment_range "16,17,18" \
    --use_cot ${USE_COT} \
    --adaptive_strategy ${ADAPTIVE_STRATEGY} \
    > cot_logs/nips_visual_rule_cot.log 2>&1 &


echo "All Rule-CoT jobs launched."
echo "Logs are saved to cot_logs/*_rule_cot.log"
