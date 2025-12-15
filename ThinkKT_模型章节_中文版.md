# 3. 模型方法

## 3.1 模型概述

ThinkKT（Thinking-based Knowledge Tracing）是一个融合多模态视觉特征和思维链推理的知识追踪模型。与传统的知识追踪方法不同，ThinkKT通过视觉大语言模型理解题目图像内容，并利用思维链（Chain-of-Thought, CoT）机制生成可解释的推理过程，从而更准确地追踪学生的知识状态变化。

模型的整体架构如图X所示，主要由三个核心模块组成：

1. **多模态题目编码器（Visual-Language Encoder）**：提取题目图像的视觉特征，并预测知识点分布。
2. **知识推理链生成器（CoT Generator）**：基于学生历史交互和当前题目，生成可解释的推理文本。
3. **知识状态追踪器（Knowledge State Tracker）**：融合多模态特征和推理链，建模学生的知识状态变化并预测答题表现。

给定一个学生的答题序列 $\{(q_1, r_1), (q_2, r_2), ..., (q_t, r_t)\}$，其中 $q_t$ 表示第 $t$ 个问题，$r_t \in \{0, 1\}$ 表示答题结果（0为错误，1为正确）。ThinkKT的目标是预测学生回答下一个问题 $q_{t+1}$ 的正确概率：

$$P(r_{t+1} = 1 | q_1, r_1, ..., q_t, r_t, q_{t+1})$$

## 3.2 多模态题目编码器

传统的知识追踪模型通常只使用问题ID（question ID）作为输入，忽略了题目本身的语义信息。在图像题目场景中，题目内容以图像形式呈现，包含题干、选项、图表等丰富的视觉信息。ThinkKT引入多模态题目编码器来提取这些视觉特征。

### 3.2.1 视觉特征提取

对于每个问题 $q_t$，我们首先获取其对应的题目图像 $I_t$。使用预训练的视觉大语言模型 Qwen2.5-VL 提取图像的视觉特征：

$$\mathbf{H}_t = \text{VisionModel}(I_t)$$

其中 $\mathbf{H}_t \in \mathbb{R}^{L \times d_v}$ 表示视觉模型的隐藏状态，$L$ 为序列长度，$d_v$ 为隐藏维度（通常为2048维）。

为了得到固定维度的题目特征表示，我们使用线性投影层将视觉特征压缩到目标维度：

$$\mathbf{v}_t = \text{LinearProj}(\text{MeanPool}(\mathbf{H}_t))$$

其中 $\mathbf{v}_t \in \mathbb{R}^{d_q}$ 是题目特征向量，$d_q = 1024$ 为题目特征维度。

### 3.2.2 知识点分布预测

除了提取视觉特征，我们还利用视觉信息预测题目涉及的知识点分布。这有助于模型理解题目的知识结构。知识点分布通过一个分类器预测：

$$\mathbf{k}_t = \sigma(\text{MLP}(\mathbf{v}_t))$$

其中 $\mathbf{k}_t \in [0,1]^{|\mathcal{C}|}$ 表示知识点概率分布，$|\mathcal{C}|$ 为知识点总数，$\sigma$ 为Sigmoid激活函数。分类器由两个全连接层组成，中间使用ReLU激活和Dropout正则化。

### 3.2.3 特征缓存机制

由于视觉特征提取计算成本较高，我们实现了特征缓存机制。对于每个问题ID，其视觉特征和知识点分布会被缓存到磁盘。在训练和推理过程中，如果缓存存在则直接加载，大大提升了模型效率。

## 3.3 知识推理链生成器

为了增强模型的可解释性和推理能力，ThinkKT引入了思维链（Chain-of-Thought, CoT）机制。CoT生成器基于学生历史交互序列和当前题目，生成可解释的推理文本，描述学生的学习状态和知识掌握情况。

### 3.3.1 推理链生成

对于时间步 $t$，给定学生历史交互序列 $\{(q_1, r_1), ..., (q_{t-1}, r_{t-1})\}$ 和当前题目 $q_t$，我们首先构建提示（prompt）模板：

$$\text{Prompt}_t = f_{\text{prompt}}(\{(q_1, r_1), ..., (q_{t-1}, r_{t-1})\}, q_t, \mathcal{K})$$

其中 $\mathcal{K}$ 为知识点词表，$f_{\text{prompt}}$ 为提示构建函数，包含以下信息：
- 学生历史答题记录
- 历史题目涉及的知识点
- 当前题目及其相关知识点

将提示和题目图像输入到多模态大语言模型（Qwen2.5-VL），生成推理链文本：

$$\text{CoT}_t = \text{MLLM}(I_t, \text{Prompt}_t)$$

推理链文本描述了学生的学习轨迹，例如："学生已掌握三角形面积公式，当前题目考查勾股定理，两者具有关联性。"

### 3.3.2 推理链编码

生成的推理链文本需要通过文本编码器转换为数值向量。我们使用预训练的Sentence-BERT模型对CoT文本进行编码：

$$\mathbf{r}_t = \text{SentenceBERT}(\text{CoT}_t)$$

其中 $\mathbf{r}_t \in \mathbb{R}^{d_c}$ 为CoT嵌入向量，$d_c = 384$ 为CoT特征维度。

### 3.3.3 缓存优化

CoT生成过程计算成本较高（需要运行大语言模型推理）。为了提高训练效率，我们实现了CoT缓存机制。对于相同的输入（相同的历史交互序列和当前题目），生成的CoT文本和嵌入向量会被缓存。在后续训练中，相同输入直接从缓存读取，避免重复计算。

## 3.4 知识状态追踪器

知识状态追踪器（ThinkKTNet）是模型的核心模块，负责融合多模态特征并建模学生的知识状态变化。

### 3.4.1 特征融合

在时间步 $t$，我们融合以下特征：
- 题目特征 $\mathbf{v}_t$：来自多模态编码器
- 答题结果嵌入 $\mathbf{a}_t$：通过嵌入层将答题结果 $r_t$ 编码为向量
- CoT嵌入 $\mathbf{r}_t$：来自推理链生成器（可选）
- 知识点分布 $\mathbf{k}_t$：来自多模态编码器

特征融合过程为：

$$\mathbf{z}_t = [\mathbf{v}_t; \mathbf{a}_t; \mathbf{k}_t]$$

如果启用CoT，则：

$$\mathbf{z}_t = [\mathbf{v}_t; \mathbf{a}_t; \mathbf{r}_t; \mathbf{k}_t]$$

其中 $[\cdot;\cdot]$ 表示向量拼接。融合后的特征通过一个融合层进行降维和归一化：

$$\mathbf{h}_t' = \text{LayerNorm}(\text{ReLU}(\text{Linear}(\mathbf{z}_t)))$$

其中 $\mathbf{h}_t' \in \mathbb{R}^{d_k}$，$d_k = 512$ 为知识状态维度。

### 3.4.2 序列建模

为了捕捉学生知识状态的时序变化，我们使用序列模型建模历史交互。支持两种序列模型：

**Transformer编码器**：使用多头自注意力机制捕捉长距离依赖关系。知识状态序列计算为：

$$\mathbf{H} = \text{TransformerEncoder}([\mathbf{h}_1', \mathbf{h}_2', ..., \mathbf{h}_t'])$$

其中每一层Transformer的计算为：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{H}^{l+1} = \text{LayerNorm}(\mathbf{H}^l + \text{FeedForward}(\text{LayerNorm}(\mathbf{H}^l + \text{Attention}(\mathbf{H}^l))))$$

**LSTM网络**：作为替代方案，也可以使用LSTM建模时序依赖：

$$(\mathbf{h}_t, \mathbf{c}_t) = \text{LSTM}(\mathbf{h}_t', \mathbf{h}_{t-1}, \mathbf{c}_{t-1})$$

其中 $\mathbf{h}_t$ 和 $\mathbf{c}_t$ 分别为隐藏状态和细胞状态。

### 3.4.3 预测层

基于建模得到的知识状态 $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_t]$，我们使用预测层计算学生答对概率：

$$\hat{y}_t = \sigma(\text{MLP}(\mathbf{h}_t))$$

其中 $\hat{y}_t \in [0,1]$ 为预测的答对概率，$\sigma$ 为Sigmoid激活函数。MLP包含两层全连接网络，中间使用ReLU激活和Dropout正则化。

此外，模型还可以预测学生对各知识点的掌握程度：

$$\mathbf{m}_t = \sigma(\text{MLP}_{\text{KC}}(\mathbf{h}_t))$$

其中 $\mathbf{m}_t \in [0,1]^{|\mathcal{C}|}$ 表示知识点掌握度向量。

## 3.5 训练目标

模型使用二元交叉熵损失进行训练。给定训练样本 $\{(q_1, r_1), ..., (q_T, r_T)\}$，损失函数定义为：

$$\mathcal{L} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \left[ r_{t+1} \log \hat{y}_t + (1-r_{t+1}) \log(1-\hat{y}_t) \right]$$

其中 $\hat{y}_t$ 为模型在时间步 $t$ 预测的答对概率，$r_{t+1}$ 为真实标签。

在训练过程中，我们使用Adam优化器进行参数更新，并采用早停（early stopping）策略防止过拟合。对于多模态编码器中的预训练模型（如Qwen2.5-VL），我们采用冻结或微调策略，根据实际效果选择合适的训练策略。

## 3.6 强化学习优化

为了进一步提升CoT生成的质量，ThinkKT引入强化学习（Reinforcement Learning, RL）机制来优化CoT生成器。该方法通过奖励信号指导模型生成更有利于知识追踪任务的高质量推理链。

### 3.6.1 奖励函数设计

强化学习的核心是设计合适的奖励函数。我们设计了一个多目标奖励函数，综合考虑预测准确性、一致性、知识点覆盖和长度约束：

$$R_t = \lambda_1 R_{\text{pred}} + \lambda_2 R_{\text{cons}} + \lambda_3 R_{\text{kc}} + \lambda_4 R_{\text{len}}$$

其中 $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 为权重超参数（默认值分别为1.0, 0.5, 0.3, 0.1）。

**（1）预测准确性奖励**：衡量CoT对预测性能的提升程度。

$$R_{\text{pred}} = \text{BCE}(\hat{y}_t^{\text{no-CoT}}, r_{t+1}) - \text{BCE}(\hat{y}_t^{\text{CoT}}, r_{t+1})$$

其中 $\hat{y}_t^{\text{no-CoT}}$ 和 $\hat{y}_t^{\text{CoT}}$ 分别表示不使用和使用CoT时的预测概率，$\text{BCE}$ 为二元交叉熵损失。该奖励鼓励生成能够提升预测准确性的CoT。

**（2）一致性奖励**：确保CoT中描述的学生状态与模型预测保持一致。如果CoT声称学生"已掌握"相关知识点，则预测概率应该较高；如果声称"薄弱"，则预测概率应该较低。

$$R_{\text{cons}} = \begin{cases}
\hat{y}_t & \text{if CoT mentions "掌握"} \\
1 - \hat{y}_t & \text{if CoT mentions "薄弱"} \\
0.5 & \text{otherwise}
\end{cases}$$

**（3）知识点覆盖奖励**：鼓励CoT提及题目的相关知识点。

$$R_{\text{kc}} = \frac{|\text{Mentioned KCs} \cap \text{True KCs}|}{|\text{True KCs}|}$$

其中 $\text{True KCs}$ 为题目实际涉及的知识点集合，$\text{Mentioned KCs}$ 为CoT中提及的知识点集合。

**（4）长度惩罚**：鼓励生成长度适中的CoT（建议在80-120 tokens之间）。

$$R_{\text{len}} = \begin{cases}
0 & \text{if } 80 \leq |\text{CoT}| \leq 120 \\
-0.1 \cdot \frac{|80 - |\text{CoT}||}{80} & \text{if } |\text{CoT}| < 80 \\
-0.1 \cdot \frac{||\text{CoT}| - 120|}{120} & \text{if } |\text{CoT}| > 120
\end{cases}$$

### 3.6.2 策略梯度优化

我们使用策略梯度方法（Policy Gradient）优化CoT生成器。在RL训练阶段，冻结知识追踪模型参数，只优化CoT生成器。采用REINFORCE算法的简化版本：

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{(\text{CoT}_t, r_{t+1}) \sim \pi_\theta} [\log \pi_\theta(\text{CoT}_t | \text{Context}_t) \cdot (R_t - b_t)]$$

其中 $\pi_\theta$ 为CoT生成策略（由MLLM参数化），$\text{Context}_t$ 表示历史交互和当前题目，$b_t$ 为基线（baseline），用于减少方差。

在实际实现中，由于直接优化大语言模型的生成过程计算成本较高，我们采用以下策略：
1. **简化优化**：主要优化CoT文本编码器的投影层参数，而非整个MLLM。
2. **代理奖励**：使用CoT嵌入向量的特征作为生成质量的代理指标。
3. **批量训练**：对每个批次计算平均奖励，并减去批次均值作为基线。

### 3.6.3 RL训练流程

RL训练分为三个阶段：

**阶段1：预训练基础模型**（基线版本）
- 使用多模态特征训练知识追踪模型
- 模型学习基本的预测能力

**阶段2：CoT生成训练**（CoT版本）
- 启用CoT生成器
- 使用固定策略生成CoT，联合训练知识追踪模型和CoT生成器
- 该阶段CoT生成策略尚未优化

**阶段3：强化学习优化**（完整版本）
- 冻结知识追踪模型参数
- 基于奖励信号优化CoT生成器
- 通过多轮迭代，逐步提升CoT生成质量

在RL训练过程中，模型会探索不同的CoT生成策略，通过奖励信号学习生成更有利于提升预测准确性和可解释性的推理链。

## 3.7 模型变体

ThinkKT支持三种模型变体，以满足不同场景的需求：

1. **基础版本（Baseline）**：仅使用多模态视觉特征，不包含CoT生成。设置参数为：`use_visual=1, use_cot=0`。

2. **CoT增强版本（CoT Version）**：使用多模态特征和思维链生成，但不进行强化学习优化。设置参数为：`use_visual=1, use_cot=1`。

3. **完整版本（Full Version）**：包含所有模块，并使用强化学习优化CoT生成质量。设置参数为：`use_visual=1, use_cot=1`，并启用RL训练。

不同的模型变体在准确性和可解释性之间提供了不同的权衡，用户可以根据实际需求选择合适的版本。

