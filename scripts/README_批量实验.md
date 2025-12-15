# ThinkKT 批量实验脚本使用指南

## 脚本功能

`run_all_thinkkt_experiments.py` 自动运行所有ThinkKT实验组合：

- **3个数据集**: DBE_KT22, XES3G5M, NIPS_task34
- **2种序列模型**: transformer, lstm  
- **3种层数**: 1, 2, 3
- **总计**: 3 × 2 × 3 = **18个实验**

每个实验包括：
1. 训练模型
2. 测试模型性能

## 快速开始

### 基础用法（Baseline版本，无CoT）

```bash
cd /home3/zhiyu/code-5/CRKT/five-thinkkt

python scripts/run_all_thinkkt_experiments.py \
    --base_dir scripts_training2testing/examples \
    --gpu_id 0 \
    --fold 0 \
    --use_cot 0 \
    --num_epochs 200 \
    --batch_size 32
```

### CoT版本（使用CoT但无RL）

```bash
python scripts/run_all_thinkkt_experiments.py \
    --base_dir scripts_training2testing/examples \
    --gpu_id 0 \
    --fold 0 \
    --use_cot 1 \
    --num_epochs 200 \
    --batch_size 32
```

## 参数说明

- `--base_dir`: 工作目录（包含wandb_thinkkt_train.py的目录）
- `--gpu_id`: 使用的GPU编号（默认"0"）
- `--fold`: 交叉验证折数（默认0）
- `--use_cot`: 是否使用CoT（0=Baseline, 1=CoT版本）
- `--num_epochs`: 训练轮数（默认200）
- `--batch_size`: 批次大小（默认32）
- `--skip_training`: 跳过训练，只运行测试（用于重新测试已训练的模型）
- `--skip_testing`: 跳过测试，只运行训练

## 实验组合

脚本会自动生成以下18个实验：

| # | 数据集 | 序列模型 | 层数 |
|---|--------|---------|------|
| 1 | DBE_KT22 | transformer | 1 |
| 2 | DBE_KT22 | transformer | 2 |
| 3 | DBE_KT22 | transformer | 3 |
| 4 | DBE_KT22 | lstm | 1 |
| 5 | DBE_KT22 | lstm | 2 |
| 6 | DBE_KT22 | lstm | 3 |
| 7 | XES3G5M | transformer | 1 |
| 8 | XES3G5M | transformer | 2 |
| 9 | XES3G5M | transformer | 3 |
| 10 | XES3G5M | lstm | 1 |
| 11 | XES3G5M | lstm | 2 |
| 12 | XES3G5M | lstm | 3 |
| 13 | NIPS_task34 | transformer | 1 |
| 14 | NIPS_task34 | transformer | 2 |
| 15 | NIPS_task34 | transformer | 3 |
| 16 | NIPS_task34 | lstm | 1 |
| 17 | NIPS_task34 | lstm | 2 |
| 18 | NIPS_task34 | lstm | 3 |

## 输出结果

### 日志文件

- **总日志**: `experiment_logs/all_experiments_[时间戳].log`
- **单个实验日志**: `experiment_logs/[实验名称]_[时间戳].log`

### 模型保存

- **Baseline版本**: `saved_model/baseline_version/[数据集]_[参数组合]/`
- **CoT版本**: `saved_model/cot_version/[数据集]_[参数组合]/`

### 测试结果

每个模型的测试结果会保存在对应的模型目录中，包括：
- `qkcs_test_predictions.txt`
- `qkcs_test_window_predictions.txt`
- 等等

## 运行时间估算

假设每个实验：
- 训练时间：2-5小时（取决于数据集大小）
- 测试时间：10-30分钟

**总计约36-90小时**（如果顺序运行）

建议：
- 使用多个GPU并行运行
- 或者分批次运行（先运行一个数据集）

## 示例：只运行一个数据集

修改脚本中的 `datasets` 列表：

```python
datasets = ["DBE_KT22"]  # 只运行DBE_KT22
```

或者创建一个简化版本，通过参数指定：

```bash
# 只运行DBE_KT22的实验（需要修改脚本添加--datasets参数）
```

## 监控进度

脚本会实时输出：
- 当前实验编号
- 训练进度
- 测试结果
- 成功/失败统计

可以查看日志文件了解详细进度。

## 故障恢复

如果某个实验失败，脚本会：
1. 记录失败的实验名称
2. 继续运行下一个实验
3. 最后输出失败列表

可以单独重新运行失败的实验，或使用 `--skip_training` 只测试已训练的模型。

## 注意事项

1. **磁盘空间**: 确保有足够的空间保存所有模型（每个模型约几百MB）
2. **GPU内存**: 确保GPU内存充足，特别是CoT版本
3. **时间**: 完整运行需要很长时间，建议在后台运行或使用screen/tmux
4. **缓存**: CoT版本首次运行会生成缓存，后续会更快

