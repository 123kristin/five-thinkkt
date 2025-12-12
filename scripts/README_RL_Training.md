# RL 训练使用说明

## 概述

RL 训练是 ThinkKT 模型的阶段2训练，用于优化 CoT 生成质量。

## 前提条件

1. **已完成阶段1训练**：需要先训练好基础 ThinkKT 模型（KT模型）
2. **数据集准备**：准备好训练数据
3. **图片路径**：确保题目图片路径正确配置

## 使用方法

```bash
python scripts/train_rl.py \
    --dataset_name XES3G5M \
    --fold 0 \
    --kt_model_path <训练好的KT模型路径> \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-6 \
    --gpu_id 0
```

## 参数说明

- `--dataset_name`: 数据集名称（DBE_KT22 或 XES3G5M）
- `--fold`: 交叉验证折数（0-4）
- `--kt_model_path`: **必需**，已训练的KT模型路径（包含 config.json 和 qkcs_model.ckpt）
- `--mllm_name`: MLLM模型路径（默认：/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct）
- `--d_cot`: CoT嵌入维度（默认：384）
- `--batch_size`: 批次大小（默认：4，RL训练建议小batch）
- `--num_epochs`: 训练轮数（默认：5）
- `--learning_rate`: 学习率（默认：1e-6）
- `--save_dir`: 模型保存目录（默认：saved_models/cot_rl）
- `--gpu_id`: GPU ID（默认：0）

## 训练流程

1. **加载KT模型**：从指定路径加载已训练的ThinkKT模型（冻结参数）
2. **初始化CoT生成器**：创建CoT生成器（可训练）
3. **初始化RL训练器**：创建RL训练器，配置奖励权重
4. **训练循环**：
   - 为每个batch生成CoT
   - 使用KT模型计算预测准确性
   - 计算奖励（预测准确性、一致性、知识点覆盖、长度惩罚）
   - 使用策略梯度更新CoT生成器参数
5. **保存模型**：保存最佳CoT生成器

## 奖励函数

RL训练使用多目标奖励函数：

- **预测准确性奖励** (权重: 1.0)：使用CoT后预测准确性提升
- **一致性奖励** (权重: 0.5)：CoT声称与预测结果的一致性
- **知识点覆盖奖励** (权重: 0.3)：CoT中提及的知识点覆盖率
- **长度惩罚** (权重: 0.1)：鼓励CoT长度在80-120 tokens之间

## 注意事项

1. **简化实现**：当前版本是简化实现，主要优化文本编码器参数。完整的RL训练需要：
   - 使用LoRA微调MLLM
   - 通过logits计算真实的log-probability
   - 或使用GRPO/PPO等序列级RL算法

2. **计算资源**：RL训练需要更多计算资源，建议使用较小的batch size

3. **训练稳定性**：RL训练可能不稳定，建议：
   - 使用较小的学习率
   - 监控奖励值的变化
   - 适当调整奖励权重

4. **模型保存**：训练过程中会保存最佳模型（基于平均奖励）

## 输出文件

- `cot_generator_epoch_<epoch>.pt`: 每个epoch的模型检查点
- `cot_cache/`: CoT缓存目录（如果启用）

## 示例

```bash
# 在XES3G5M数据集上训练，使用fold 0的KT模型
python scripts/train_rl.py \
    --dataset_name XES3G5M \
    --fold 0 \
    --kt_model_path saved_model/XES3G5M_0_0.0001_32_thinkkt_qkcs_... \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-6 \
    --gpu_id 0
```

