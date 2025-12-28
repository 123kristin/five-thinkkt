
import os
import re
import glob
import numpy as np
import pandas as pd

# 配置
# 脚本位于 bash_script/, saved_model 位于上级目录
base_dir = "../saved_model/bs"
datasets = ["XES3G5M", "DBE_KT22", "nips_task34"]
modes = ["qid", "visual", "vq", "gf", "ca", "cl"]
folds = [0, 1, 2, 3, 4]

# 正则匹配预测结果 (包含 Window Metrics)
# 示例: testauc: 0.8060..., testacc: 0.8038..., window_testauc: 0.8060..., window_testacc: 0.8039...
res_pattern = re.compile(
    r"testauc:\s+([0-9.]+),\s+testacc:\s+([0-9.]+),\s+window_testauc:\s+([0-9.]+),\s+window_testacc:\s+([0-9.]+)"
)

results = []

# 表头格式化
header = f"{'Mode':<8} {'Dataset':<12} {'Fold':<4} {'AUC':<8} {'ACC':<8} {'WinAUC':<8} {'WinACC':<8}"
print(header)
print("-" * 75)

for mode in modes:
    save_type_dir = os.path.join(base_dir, mode)
    if not os.path.exists(save_type_dir):
        continue
        
    for dataset in datasets:
        # 存储所有Fold的指标列表
        metrics = {
            "testauc": [],
            "testacc": [],
            "window_testauc": [],
            "window_testacc": []
        }
        
        for fold in folds:
            # 1. 寻找对应的 Checkpoint 目录
            # 模式: saved_model/bs/{mode}/*{dataset}*{fold}*/predicting*.log
            # 注意: fold 匹配需要小心，防止 matching fold10 with fold1
            # 但这里我们先glob目录，再看log
            
            # 使用 glob 查找可能的目录
            # 假设目录名包含 dataset 和 fold
            search_pattern = os.path.join(save_type_dir, f"*{dataset}*{fold}*")
            possible_dirs = glob.glob(search_pattern)
            
            # 过滤目录 (确保是目录且确实包含fold数字，防止'1'匹配'10')
            # 这里的fold是int 0-4，比较安全，因为通常是 _{fold}_ 或者 _{fold}
            valid_dirs = []
            for d in possible_dirs:
                if os.path.isdir(d):
                    # 双重检查 fold
                    # 简单办法：既然glob已经过滤了，而且Fold只是0-4，冲突概率小
                    # 最大的风险是 fold=1 匹配 fold=1，但也可能匹配 ..._10_...
                    # 我们假设 _fold_ 或者 _{fold}_ 格式，或者简单信任 glob，因为之前脚本也是这么找的
                    
                    # 寻找该目录下的 predicting*.log
                    log_files = glob.glob(os.path.join(d, "predicting*.log"))
                    if log_files:
                        # 找到最新的一个 log (如果有多份)
                        latest_log = max(log_files, key=os.path.getmtime)
                        valid_dirs.append(latest_log)
            
            vals = None
            if valid_dirs:
                # 如果有多个目录匹配（不太可能，除非多次实验），取最新的一个 log
                target_log = max(valid_dirs, key=os.path.getmtime)
                
                try:
                    with open(target_log, 'r') as f:
                        content = f.read()
                        matches = res_pattern.findall(content)
                        if matches:
                            # 取最后一个匹配项
                            vals = matches[-1] # (auc, acc, w_auc, w_acc)
                except Exception as e:
                    print(f"Error reading {target_log}: {e}")

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
            print(f"{mode:<8} {dataset:<12} AVG  {avg_str_list[0]} {avg_str_list[1]} (Win: {avg_str_list[2]} / {avg_str_list[3]})")
            print("-" * 75)
            
            results.append(res_row)

# 保存到 CSV
df = pd.DataFrame(results)
cols = ["Mode", "Dataset", "Fold", "AUC", "ACC", "WindowAUC", "WindowACC"]

if not df.empty:
    df = df.reindex(columns=cols)
else:
    df = pd.DataFrame(columns=cols)

df.to_csv("experiment_results_summary.csv", index=False)
print("\nResults saved to experiment_results_summary.csv")
