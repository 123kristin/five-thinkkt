# ThinkKT 文件结构

```
scripts_training2testing/examples/pykt/models/our_model/
├── thinkkt.py                          # 主模型类：整合所有模块，实现pykt标准接口
├── thinkkt_net.py                      # 知识状态追踪器：Transformer/LSTM序列建模，预测答对概率
├── visual_language_encoder.py          # 多模态编码器：使用Qwen2.5-VL提取图像特征，预测知识点分布
│
├── cot/                                # CoT模块
│   ├── cot_generator.py                # CoT生成器：使用MLLM生成推理链文本，编码为向量
│   └── cot_prompts.py                  # Prompt模板：构建CoT生成提示词，解析和验证CoT
│
└── rl/                                 # RL模块
    └── cot_rl_trainer.py              # RL训练器：多目标奖励函数，策略梯度优化CoT质量

scripts/
├── train_rl.py                         # RL训练脚本：加载KT模型，训练CoT生成器
├── precompute_question_features.py     # 特征预计算：批量提取题目特征并缓存
├── precompute_cot.py                   # CoT预生成：批量生成CoT并缓存
├── train_sft.py                        # SFT训练脚本：监督微调CoT生成器（框架）
└── README_RL_Training.md               # RL训练使用说明

scripts_training2testing/examples/
└── wandb_thinkkt_train.py              # ThinkKT训练入口：主训练脚本，支持所有配置参数
```

## 模块功能说明

| 模块/文件 | 功能 |
|-----------|------|
| **thinkkt.py** | 主模型类，整合VisualEncoder、CoTGenerator、ThinkKTNet |
| **thinkkt_net.py** | 知识状态追踪器，融合特征→序列建模→预测 |
| **visual_language_encoder.py** | 多模态编码器，图像特征提取+知识点预测 |
| **cot_generator.py** | CoT生成器，MLLM生成推理链+文本编码 |
| **cot_prompts.py** | CoT Prompt模板，构建提示词+解析响应 |
| **cot_rl_trainer.py** | RL训练器，计算奖励+策略梯度优化 |
| **train_rl.py** | RL训练脚本，完整训练流程 |

