# ThinkKT 模型使用说明

## 概述

ThinkKT 是一个多模态知识追踪模型，结合了视觉特征和思维链推理。当前版本实现了多模态编码器模块，可以提取题目图片的视觉特征并预测知识点分布。

## 文件结构

```
pykt/models/our_model/
├── thinkkt.py                    # 主模型类
├── thinkkt_net.py                # 知识状态追踪器
├── visual_language_encoder.py    # 多模态编码器
└── README_ThinkKT.md             # 本文件
```

## 已实现功能

### ✅ 多模态编码器（Visual-Language Encoder）
- 使用 Qwen2.5-VL-3B-Instruct 提取图像特征
- 预测知识点分布
- 特征缓存机制（避免重复计算）

### ✅ 知识状态追踪器（ThinkKTNet）
- Transformer 或 LSTM 序列建模
- 融合题目特征、答题结果、知识点分布
- 预测答对概率

### ✅ 主模型类（ThinkKT）
- 实现 pykt 标准接口：`train_one_step()` 和 `predict_one_step()`
- 已注册到 pykt 框架

## 使用方法

### 1. 配置模型参数

在 `my_configs/kt_config.json` 中已添加 ThinkKT 配置：

```json
{
    "thinkkt": {
        "learning_rate": 1e-4,
        "d_question": 1024,
        "d_cot": 384,
        "d_knowledge": 512,
        "dropout": 0.1,
        "seq_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "mllm_name": "Qwen/Qwen2-VL-3B-Instruct",
        "use_cot": false,
        "use_visual": true,
        "cache_dir": "features"
    }
}
```

### 2. 训练模型

**方法一：使用 ThinkKT 专用训练脚本（推荐）**

```bash
cd scripts_training2testing/examples
python wandb_thinkkt_train.py \
    --dataset_name DBE_KT22 \
    --fold 0 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_epochs 200 \
    --gpu_id 0 \
    --use_visual 1 \
    --use_cot 0
```

**方法二：使用通用训练脚本**

```bash
cd scripts_training2testing/examples
python wandb_train.py \
    --dataset_name DBE_KT22 \
    --model_name thinkkt \
    --fold 0 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_epochs 200 \
    --gpu_id 0 \
    --d_question 1024 \
    --d_knowledge 512 \
    --use_visual 1 \
    --use_cot 0
```

**ThinkKT 专用脚本的优势：**
- 包含所有 ThinkKT 特定参数的默认值
- 更清晰的参数说明
- 自动设置模型名称为 "thinkkt"
- 更友好的输出信息

### 3. 特征缓存

模型会自动缓存题目特征到 `features/{dataset_name}_question_features.pt`，避免重复计算。

首次运行时会：
1. 加载 Qwen2.5-VL 模型（可能需要一些时间）
2. 提取所有题目图片的特征
3. 保存到缓存文件

后续运行时会直接从缓存加载，大大加快训练速度。

## 配置参数说明

### 基础参数（在 `kt_config.json` 中配置，或通过命令行覆盖）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_question` | int | 1024 | 题目特征维度 |
| `d_cot` | int | 384 | CoT嵌入维度（当前未使用） |
| `d_knowledge` | int | 512 | 知识状态维度 |
| `dropout` | float | 0.1 | Dropout率 |
| `seq_model_type` | str | "transformer" | 序列模型类型："transformer" 或 "lstm" |
| `num_transformer_layers` | int | 6 | Transformer层数 |
| `num_heads` | int | 8 | 注意力头数 |
| `num_lstm_layers` | int | 2 | LSTM层数（当seq_model_type=lstm时使用） |
| `mllm_name` | str | "Qwen/Qwen2-VL-3B-Instruct" | 视觉模型路径 |
| `use_cot` | int | 0 | 是否使用CoT（0/1，当前版本为0） |
| `use_visual` | int | 1 | 是否使用视觉特征（0/1） |
| `cache_dir` | str | "features" | 特征缓存目录 |

**注意**：使用 `wandb_thinkkt_train.py` 时，所有参数都可以通过命令行覆盖，例如：

```bash
python wandb_thinkkt_train.py \
    --d_question 512 \
    --d_knowledge 256 \
    --seq_model_type lstm \
    --num_lstm_layers 3 \
    --dropout 0.2 \
    --use_visual 1
```

## 注意事项

1. **首次运行较慢**：首次运行时需要下载和加载 Qwen2.5-VL 模型，可能需要较长时间。

2. **显存需求**：Qwen2.5-VL-3B 模型需要一定的显存，建议使用至少 8GB 显存的 GPU。

3. **图片路径**：确保数据集的 `q_imgs` 目录存在且包含所有题目图片。

4. **CoT功能**：当前版本 `use_cot=false`，CoT 生成器将在后续版本实现。

5. **特征缓存**：如果修改了视觉模型或图片，需要删除缓存文件重新计算。

## 后续开发计划

- [ ] 实现 CoT 生成器
- [ ] 实现强化学习优化器
- [ ] 添加几何特征提取（可选）
- [ ] 优化特征缓存机制
- [ ] 添加更多评估指标

## 故障排除

### 问题1：无法导入视觉处理器

**错误**：`ImportError: 无法导入 QwenVisionProcessor`

**解决**：确保 `prepare_q_img_for_kt_dataset` 目录在 Python 路径中，或修改 `visual_language_encoder.py` 中的导入路径。

### 问题2：找不到图片路径

**错误**：`警告: 无法找到DBE_KT22的q_imgs目录`

**解决**：检查数据配置中的路径，或手动修改 `build_img_path_dict()` 函数中的路径。

### 问题3：显存不足

**错误**：`CUDA out of memory`

**解决**：
- 减小 batch_size
- 使用更小的视觉模型
- 启用特征缓存（避免重复计算）

## 联系方式

如有问题或建议，请查看 `ThinkKT_Implementation_Analysis.md` 获取更多实现细节。

