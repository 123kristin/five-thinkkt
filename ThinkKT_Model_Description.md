# ThinkKT: A Multimodal Knowledge Tracing Model with Chain-of-Thought Reasoning

## 1. 模型概述 (Model Overview)

ThinkKT (Thinking Knowledge Tracing) 是一个创新的**多模态知识追踪模型**，旨在通过融合视觉信息、历史交互数据和思维链推理来准确追踪学生的学习状态。该模型将知识追踪问题重新定义为一个多模态理解与推理任务，特别适用于包含图像信息的题目（如几何题、图表题等）。

**核心创新点：**
1. **多模态题目理解**：利用视觉-语言模型提取题目的视觉和文本特征
2. **知识推理链生成**：使用大语言模型生成解释性的思维链（Chain-of-Thought）
3. **强化学习优化**：通过RL优化思维链质量，使其真正有助于预测准确性
4. **端到端训练**：从题目理解到知识状态追踪的完整流程

---

## 2. 模型架构 (Architecture)

ThinkKT 由四个核心模块组成：

### 2.1 模块一：多模态题目理解器 (Multimodal Question Understander)

**功能**：将题目图片转换为可理解的向量表示

**技术细节**：
- **视觉编码器**：使用 Qwen2.5-VL (Vision-Language Model) 提取图像特征
  - 输入：题目图片（包含题干、选项、几何图形等）
  - 处理：通过预训练的视觉-语言模型提取多模态特征
  - 输出：题目特征向量 $v_t \in \mathbb{R}^{d_q}$，其中 $d_q$ 为题目特征维度（默认1024）

- **知识点预测头**：从题目特征预测相关知识点分布
  - 结构：MLP (1024 → 512 → num_c)
  - 输出：知识点分布 $k_t \in [0,1]^{|C|}$，表示题目涉及各知识点的概率

**数学表示**：
$$v_t = \text{VisualEncoder}(I_t)$$
$$k_t = \text{Sigmoid}(\text{MLP}(v_t))$$

其中 $I_t$ 为第 $t$ 个题目的图片。

---

### 2.2 模块二：知识推理链生成器 (Knowledge Chain-of-Thought Generator)

**功能**：生成解释性的思维链，说明学生可能掌握或薄弱的知识点

**技术细节**：
- **生成模型**：使用 Qwen2.5-VL 作为大语言模型生成CoT
- **输入**：
  - 历史交互序列：$\{(q_1, a_1), (q_2, a_2), ..., (q_{t-1}, a_{t-1})\}$
  - 当前题目图片：$I_t$
  - 知识点信息：历史题目和当前题目的知识点标注
- **输出**：思维链文本 $r_t$（字符串），描述学生的学习状态推理过程

**Prompt模板结构**：
1. 题目考察知识点识别
2. 学生历史掌握情况分析
3. 图像关键信息提取
4. 可能错误原因分析
5. 预测置信度

**CoT编码**：使用 Sentence Transformer 将文本CoT编码为向量
$$r_t = \text{TextEncoder}(r_t^{\text{text}}) \in \mathbb{R}^{d_c}$$
其中 $d_c$ 为CoT嵌入维度（默认384）。

---

### 2.3 模块三：知识状态追踪器 (Knowledge State Tracker)

**功能**：融合多模态特征，建模知识状态变化，预测答题概率

**技术细节**：

**步骤1：特征融合**
- 输入特征：
  - 题目特征：$v_t \in \mathbb{R}^{d_q}$
  - 答题结果嵌入：$a_t \in \mathbb{R}^{d_a}$（通过Embedding层编码0/1）
  - 知识点分布：$k_t \in [0,1]^{|C|}$
  - CoT嵌入：$r_t \in \mathbb{R}^{d_c}$（可选）

- 融合过程：
$$z_t = \text{Concat}([v_t; a_t; k_t; r_t]) \in \mathbb{R}^{d_{\text{in}}}$$
$$h_t = \text{FusionLayer}(z_t) \in \mathbb{R}^{d_k}$$

其中 FusionLayer 包括：Linear + LayerNorm + ReLU + Dropout

**步骤2：序列建模**
- **Transformer版本**：
$$H = \text{TransformerEncoder}([h_1, h_2, ..., h_T])$$
- **LSTM版本**：
$$H, (h_T, c_T) = \text{LSTM}([h_1, h_2, ..., h_T])$$

**步骤3：预测**
- 答对概率预测：
$$\hat{y}_t = \text{Sigmoid}(\text{MLP}(h_t)) \in [0,1]$$

- 知识点掌握度预测（可选，用于可解释性）：
$$m_t = \text{Sigmoid}(\text{MLP}(h_t)) \in [0,1]^{|C|}$$

**损失函数**：
$$\mathcal{L} = \text{BCE}(\hat{y}_t, y_t)$$

---

### 2.4 模块四：强化学习优化器 (RL-based Reasoning Reward)

**功能**：优化CoT生成质量，使其真正有助于知识追踪准确性

**技术细节**：

**奖励函数设计**：
$$R_{\text{total}} = \lambda_1 R_{\text{pred}} + \lambda_2 R_{\text{cons}} + \lambda_3 R_{\text{kc}} + \lambda_4 R_{\text{len}}$$

1. **预测准确性奖励** ($R_{\text{pred}}$):
$$R_{\text{pred}} = \text{BCE}(y^{\text{no-cot}}, y) - \text{BCE}(y^{\text{with-cot}}, y)$$
衡量使用CoT后预测准确性的提升

2. **一致性奖励** ($R_{\text{cons}}$):
$$R_{\text{cons}} = \begin{cases}
\hat{y}_t & \text{if CoT claims mastery} \\
1 - \hat{y}_t & \text{if CoT claims weakness} \\
0.5 & \text{otherwise}
\end{cases}$$
确保CoT声称与预测结果一致

3. **知识点覆盖奖励** ($R_{\text{kc}}$):
$$R_{\text{kc}} = \frac{|\text{KC}_{\text{mentioned}} \cap \text{KC}_{\text{label}}|}{|\text{KC}_{\text{label}}|}$$
鼓励CoT提及题目标注的知识点

4. **长度惩罚** ($R_{\text{len}}$):
$$R_{\text{len}} = \begin{cases}
0 & \text{if } 80 \leq |r_t| \leq 120 \\
-0.1 \cdot \frac{80 - |r_t|}{80} & \text{if } |r_t| < 80 \\
-0.1 \cdot \frac{|r_t| - 120}{120} & \text{if } |r_t| > 120
\end{cases}$$
鼓励CoT长度在合理范围

**训练策略**：使用 REINFORCE 策略梯度算法
$$\mathcal{L}_{\text{RL}} = -\mathbb{E}[\log \pi(r_t | s_t) \cdot (R_t - \bar{R})]$$

其中 $\pi$ 为CoT生成策略，$s_t$ 为当前状态，$\bar{R}$ 为基线奖励。

---

## 3. 训练流程 (Training Pipeline)

ThinkKT 采用**三阶段训练策略**：

### 阶段1：基础训练 (Baseline Training)
- **目标**：训练多模态编码器和知识状态追踪器
- **配置**：`use_cot=0`
- **数据**：题目图片 + 答题历史
- **损失**：二元交叉熵损失

### 阶段2：CoT增强训练 (CoT-Enhanced Training)
- **目标**：集成CoT生成器，训练端到端模型
- **配置**：`use_cot=1`
- **数据**：题目图片 + 答题历史 + 自动生成的CoT
- **损失**：二元交叉熵损失

### 阶段3：RL优化训练 (RL Optimization)
- **目标**：优化CoT生成质量
- **配置**：冻结KT模型，只训练CoT生成器
- **方法**：使用多目标奖励函数进行强化学习
- **损失**：策略梯度损失

---

## 4. 模型特点 (Key Features)

### 4.1 多模态融合
- **视觉理解**：能够理解题目中的几何图形、图表、公式等视觉信息
- **文本理解**：理解题干、选项的语义
- **联合表示**：通过Vision-Language模型实现视觉和文本的深度融合

### 4.2 可解释性
- **思维链生成**：生成人类可读的推理过程
- **知识点关联**：明确标注题目涉及的知识点
- **掌握度追踪**：输出学生对每个知识点的掌握程度

### 4.3 自适应优化
- **强化学习**：根据预测准确性动态优化CoT质量
- **多目标奖励**：平衡准确性、一致性、覆盖度等多个目标
- **端到端学习**：从题目理解到预测的完整优化

---

## 5. 数学形式化 (Mathematical Formulation)

给定学生的学习历史 $\mathcal{H} = \{(q_1, a_1, I_1), ..., (q_{t-1}, a_{t-1}, I_{t-1})\}$ 和当前题目 $(q_t, I_t)$，ThinkKT 的目标是预测学生答对的概率：

$$\hat{y}_t = P(a_t = 1 | \mathcal{H}, q_t, I_t)$$

**模型计算过程**：

1. **题目特征提取**：
$$v_t = f_{\text{visual}}(I_t), \quad k_t = f_{\text{kc}}(v_t)$$

2. **思维链生成**（如果启用）：
$$r_t = f_{\text{cot}}(\mathcal{H}, I_t), \quad r_t^{\text{emb}} = f_{\text{encode}}(r_t)$$

3. **知识状态更新**：
$$h_t = f_{\text{tracker}}(v_t, a_{t-1}, k_t, r_t^{\text{emb}}, h_{t-1})$$

4. **预测**：
$$\hat{y}_t = f_{\text{predictor}}(h_t)$$

---

## 6. 技术实现细节 (Implementation Details)

### 6.1 特征缓存机制
- **目的**：避免重复计算题目特征，加速训练
- **实现**：题目ID → 特征向量的映射缓存
- **存储**：PyTorch tensor格式，支持持久化

### 6.2 批处理优化
- **视觉编码**：支持批量图片处理
- **CoT生成**：支持批量生成，但受限于MLLM的计算资源
- **序列建模**：使用Transformer/LSTM的批处理能力

### 6.3 设备管理
- **多GPU支持**：通过环境变量指定GPU
- **混合精度**：支持FP16/BF16推理加速
- **内存优化**：特征缓存、梯度检查点等技术

---

## 7. 适用场景 (Application Scenarios)

ThinkKT 特别适用于以下场景：

1. **几何题目**：包含图形、标注的几何题
2. **图表题目**：涉及数据可视化、图表的题目
3. **综合题目**：需要同时理解文字和图像的复合题目
4. **多步骤推理**：需要推理链解释的复杂题目

---

## 8. 与现有方法的对比

| 特性 | 传统KT模型 | ThinkKT |
|------|-----------|---------|
| 输入模态 | 文本/ID | **多模态（图像+文本）** |
| 题目理解 | 简单嵌入 | **视觉-语言模型** |
| 可解释性 | 弱 | **思维链生成** |
| 优化方式 | 监督学习 | **监督学习+强化学习** |
| 知识建模 | 隐式 | **显式+隐式** |

---

## 9. 总结

ThinkKT 是一个**端到端的多模态知识追踪框架**，通过融合视觉理解、思维链推理和强化学习，实现了更准确、更可解释的知识追踪。该模型特别适合处理包含图像信息的题目，能够提供人类可理解的推理过程，并通过强化学习不断优化推理质量。

**核心贡献**：
1. 将知识追踪扩展到多模态领域
2. 引入思维链提升可解释性
3. 使用强化学习优化推理质量
4. 端到端的多阶段训练策略

