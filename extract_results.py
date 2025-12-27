
import os
import re
import numpy as np
import pandas as pd

# 配置
log_dir = "saved_model/bs/logs"
datasets = ["XES3G5M", "DBE_KT22", "nips_task34"]
modes = ["qid", "visual", "vq"]
folds = [0, 1, 2, 3, 4]

# 正则匹配预测结果 (包含 Window Metrics)
# 示例: testauc: 0.8060..., testacc: 0.8038..., window_testauc: 0.8060..., window_testacc: 0.8039...
res_pattern = re.compile(
    r"testauc:\s+([0-9.]+),\s+testacc:\s+([0-9.]+),\s+window_testauc:\s+([0-9.-]+),\s+window_testacc:\s+([0-9.-]+)"
)

results = []

# 表头格式化
header = f"{'Mode':<8} {'Dataset':<12} {'Fold':<4} {'AUC':<8} {'ACC':<8} {'WinAUC':<8} {'WinACC':<8}"
print(header)
print("-" * 75)

for mode in modes:
    for dataset in datasets:
        # 存储所有Fold的指标列表
        metrics = {
            "testauc": [],
            "testacc": [],
            "window_testauc": [],
            "window_testacc": []
        }
        
        for fold in folds:
            log_file = os.path.join(log_dir, f"{mode}_{dataset}_fold{fold}.log")
            
            vals = None
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    matches = res_pattern.findall(content)
                    if matches:
                        # 取最后一个匹配项
                        vals = matches[-1] # (auc, acc, w_auc, w_acc)
            
            if vals:
                auc, acc, w_auc, w_acc = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                
                # 打印单折结果
                print(f"{mode:<8} {dataset:<12} {fold:<4} {auc:.4f}   {acc:.4f}   {w_auc:.4f}   {w_acc:.4f}")
                
                metrics["testauc"].append(auc)
                metrics["testacc"].append(acc)
                metrics["window_testauc"].append(w_auc)
                metrics["window_testacc"].append(w_acc)
                
                results.append({
                    "Mode": mode,
                    "Dataset": dataset,
                    "Fold": fold,
                    "AUC": auc,
                    "ACC": acc,
                    "WindowAUC": w_auc,
                    "WindowACC": w_acc
                })
            else:
                print(f"{mode:<8} {dataset:<12} {fold:<4} MISSING/FAIL")
        
        # 计算平均值和标准差 (只要有数据)
        if metrics["testauc"]:
            res_row = {
                "Mode": mode,
                "Dataset": dataset,
                "Fold": "AVG"
            }
            
            avg_str_list = []
            
            # 依次计算4个指标
            for k, csv_k in [('testauc','AUC'), ('testacc','ACC'), ('window_testauc','WindowAUC'), ('window_testacc','WindowACC')]:
                data = metrics[k]
                # 过滤掉可能的 -1 (如果脚本输出-1代表无数据)
                valid_data = [x for x in data if x != -1]
                
                if valid_data:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    mean_std_str = f"{mean_val:.4f}±{std_val:.4f}"
                    res_row[csv_k] = mean_std_str
                    avg_str_list.append(mean_std_str)
                else:
                    res_row[csv_k] = "N/A"
                    avg_str_list.append("N/A")

            # 打印平均行
            # 格式化输出稍微紧凑一点以适应两列
            print(f"{mode:<8} {dataset:<12} AVG  {avg_str_list[0]} {avg_str_list[1]} (Win: {avg_str_list[2]} / {avg_str_list[3]})")
            print("-" * 75)
            
            results.append(res_row)

# 保存到 CSV
df = pd.DataFrame(results)
cols = ["Mode", "Dataset", "Fold", "AUC", "ACC", "WindowAUC", "WindowACC"]

if not df.empty:
    # 调整列顺序
    # 确保所有列都在 df 中 (对于 missing 的行，可能只会在 df 中产生 partial columns 如果没有一行是完整的)
    # 为安全起见，我们重新索引
    df = df.reindex(columns=cols)
else:
    # 创建空表
    df = pd.DataFrame(columns=cols)

df.to_csv("experiment_results_summary.csv", index=False)
print("\nResults saved to experiment_results_summary.csv")
