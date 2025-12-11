# DKT模型数学公式描述

本文档描述了增强版DKT (Deep Knowledge Tracing) 模型中各个组件的数学公式。

## 1. 基础DKT模型

### 1.1 交互嵌入计算
```
x = q + num_q * r
xemb = Embedding(x)
```
其中：
- `q`: 习题ID序列 (batch_size, seq_len)
- `r`: 答题结果序列 (batch_size, seq_len) 
- `num_q`: 习题总数
- `xemb`: 交互嵌入 (batch_size, seq_len, emb_size)

### 1.2 LSTM序列建模
```
h_t, (h_n, c_n) = LSTM(h_{t-1}, x_t)
```
其中：
- `h_t`: 隐藏状态 (batch_size, seq_len, hidden_size)
- `x_t`: 输入特征 (batch_size, seq_len, emb_size)

### 1.3 输出预测
```
y_t = sigmoid(W_out * h_t)
```
其中：
- `y_t`: 预测概率 (batch_size, seq_len, num_concepts)
- `W_out`: 输出权重矩阵

## 2. 交叉注意力机制

### 2.1 基础交叉注意力块
```
1. 投影层:
   Q = W_q * content_emb
   K = W_k * [analysis_emb; kc_emb]
   V = W_v * [analysis_emb; kc_emb]

2. 多头注意力:
   Attention(Q,K,V) = softmax(QK^T/√d_k)V

3. 残差连接和层归一化:
   output = LayerNorm(dropout(W_o * Attention(Q,K,V)) + Q)
```

### 2.2 多层交叉注意力
```
1. 输入投影:
   x_content = W_c * content_emb
   x_analysis = W_a * analysis_emb  
   x_kc = W_k * kc_emb

2. 逐层处理 (l = 1,2,...,L):
   h_l = CrossAttention(h_{l-1}, x_analysis, x_kc)
   h_0 = x_content

3. 最终输出:
   output = LayerNorm(dropout(W_f * h_L) + h_L)
```

### 2.3 改进交叉注意力（双向+门控）
```
1. 输入投影:
   x_content = W_c * content_emb
   x_analysis = W_a * analysis_emb
   x_kc = W_k * kc_emb

2. 双向交叉注意力 (l = 1,2,...,L):
   content_to_others = CrossAttention(x_content, x_analysis, x_kc)
   others_to_content = CrossAttention(x_analysis, x_content, x_kc)

3. 门控融合:
   gate = sigmoid(W_g * [content_to_others; others_to_content])
   x_content = gate * content_to_others + (1 - gate) * others_to_content

4. 最终输出:
   output = LayerNorm(dropout(W_f * x_content) + x_content)
```

### 2.4 分层注意力
```
1. 输入投影:
   x_content = W_c * content_emb
   x_analysis = W_a * analysis_emb
   x_kc = W_k * kc_emb

2. 同模态自注意力 (l = 1,2,...,L):
   content_self = SelfAttention(x_content)
   analysis_self = SelfAttention(x_analysis)
   kc_self = SelfAttention(x_kc)

3. 跨模态交叉注意力:
   enhanced_content = CrossAttention(content_self, analysis_self, kc_self)

4. 最终输出:
   output = LayerNorm(dropout(W_f * enhanced_content) + enhanced_content)
```

### 2.5 知识感知注意力
```
1. 输入投影:
   x_content = W_c * content_emb
   x_analysis = W_a * analysis_emb
   x_kc = W_k * kc_emb

2. 知识权重计算:
   knowledge_weights = sigmoid(W_kw * x_kc)

3. 知识感知的嵌入增强:
   content_enhanced = x_content * knowledge_weights
   analysis_enhanced = x_analysis * knowledge_weights

4. 知识感知交叉注意力 (l = 1,2,...,L):
   h_l = CrossAttention(h_{l-1}, analysis_enhanced, x_kc)
   h_0 = content_enhanced

5. 最终输出:
   output = LayerNorm(dropout(W_f * h_L) + h_L)
```

## 3. 预训练嵌入融合

### 3.1 嵌入获取
```
valid_qids = clamp(qids, 0, emb_size - 1)
content_emb = content_emb_data[valid_qids]
analysis_emb = analysis_emb_data[valid_qids]
kc_emb = kc_emb_data[valid_qids]
```

### 3.2 交叉注意力融合
```
enhanced_content = CrossAttention(content_emb, analysis_emb, kc_emb)
```

### 3.3 最终融合
```
if no_analysis_fusion:
    fused_emb = W_fusion * enhanced_content
else:
    combined_emb = [enhanced_content; analysis_emb]
    fused_emb = W_fusion * combined_emb
```

## 4. 完整模型前向传播

### 4.1 基本流程
```
1. 交互嵌入计算:
   x = q + num_q * r
   xemb = Embedding(x)

2. 预训练嵌入获取 (如果启用):
   pretrain_emb = get_pretrain_emb(q)

3. 特征融合:
   if pretrain_emb is not None:
       combined_features = [xemb; pretrain_emb]
       h = W_fusion * combined_features
   else:
       h = xemb

4. LSTM序列建模:
   h_t, (h_n, c_n) = LSTM(h_{t-1}, h_t)

5. 输出预测:
   h = dropout(h)
   y = sigmoid(W_out * h)
```

## 5. 训练目标

### 5.1 损失函数（二元交叉熵）
```
L = -Σ_{t=1}^T Σ_{c=1}^C [r_{t,c} * log(y_{t,c}) + (1-r_{t,c}) * log(1-y_{t,c})]
```
其中：
- `r_{t,c}`: 时间步t概念c的真实掌握情况 (0或1)
- `y_{t,c}`: 时间步t概念c的预测掌握概率 [0,1]
- `T`: 序列长度
- `C`: 概念数量

### 5.2 优化目标
```
min_θ L(θ) + λ||θ||_2^2
```
其中：
- `θ`: 模型参数
- `λ`: L2正则化系数
- `||θ||_2^2`: L2正则化项

## 6. 评估指标

### 6.1 AUC (Area Under the ROC Curve)
ROC曲线下的面积，用于评估二分类模型的性能。

### 6.2 ACC (Accuracy)
```
ACC = (TP + TN) / (TP + TN + FP + FN)
```
其中：
- `TP`: 真阳性 (True Positive)
- `TN`: 真阴性 (True Negative)  
- `FP`: 假阳性 (False Positive)
- `FN`: 假阴性 (False Negative)

## 7. 模型参数说明

### 7.1 核心参数
- `num_c`: 概念数量
- `emb_size`: 嵌入维度
- `hidden_size`: LSTM隐藏层维度
- `dropout`: Dropout比率

### 7.2 注意力参数
- `d_model`: 注意力模型维度
- `num_heads`: 多头注意力头数
- `num_layers`: 注意力层数
- `attention_type`: 注意力类型

### 7.3 嵌入参数
- `content_dim`: 内容嵌入维度
- `analysis_dim`: 解析嵌入维度
- `kc_dim`: 知识概念嵌入维度
- `trainable_*_emb`: 嵌入是否可训练

## 8. 实验配置

### 8.1 实验类型
- `basic`: 基本DKT模型
- `enhanced_fixed`: 增强DKT模型（嵌入固定）
- `enhanced_trainable`: 增强DKT模型（嵌入可训练）
- `enhanced_no_analysis`: 增强DKT模型（不拼接解析嵌入）
- `enhanced_trainable_no_analysis`: 增强DKT模型（嵌入可训练且不拼接解析嵌入）

### 8.2 注意力类型
- `cross`: 原始交叉注意力
- `improved_cross`: 改进交叉注意力
- `hierarchical`: 分层注意力
- `knowledge_aware`: 知识感知注意力 