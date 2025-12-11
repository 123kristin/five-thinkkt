# ThinkKT 模型实现需求分析

基于 pykt 库架构的 ThinkKT 实现方案

## 一、pykt 库核心接口理解

### 1.1 模型接口规范

**必须实现的接口：**
- `train_one_step(data) -> (y, loss)`: 训练一步，返回预测和损失
- `predict_one_step(data) -> y`: 预测一步，返回预测结果
- `model_name`: 模型名称字符串属性
- `get_loss(ys, rshft, sm) -> loss`: 计算损失（可选，可在 train_one_step 中直接计算）

**数据字典格式（data）：**
```python
{
    'qseqs': torch.Tensor,      # (batch, seq_len-1) 问题ID序列（前seq_len-1个）
    'cseqs': torch.Tensor,      # (batch, seq_len-1, max_concepts) 知识点序列
    'rseqs': torch.Tensor,      # (batch, seq_len-1) 答题结果序列（前seq_len-1个）
    'shft_qseqs': torch.Tensor, # (batch, seq_len-1) 问题ID序列（后seq_len-1个，用于预测）
    'shft_cseqs': torch.Tensor, # (batch, seq_len-1, max_concepts) 知识点序列（后seq_len-1个）
    'shft_rseqs': torch.Tensor, # (batch, seq_len-1) 答题结果序列（后seq_len-1个，用于标签）
    'smasks': torch.Tensor,     # (batch, seq_len-1) 选择掩码，用于计算loss
    'masks': torch.Tensor       # (batch, seq_len-1) 掩码序列
}
```

### 1.2 模型注册机制

在 `pykt/models/init_model.py` 中需要添加 ThinkKT 的初始化逻辑：
```python
elif model_name == "thinkkt":
    model = ThinkKT(data_config["num_q"], data_config["num_c"], 
                     **model_config, emb_type=emb_type).to(device)
```

### 1.3 训练流程

1. **数据加载**: `init_dataset4train()` 返回 `train_loader, valid_loader`
2. **模型初始化**: `init_model()` 根据模型名创建模型实例
3. **训练循环**: `train_model()` 中调用 `model.train_one_step(data)`
4. **评估**: 调用 `model.predict_one_step(data)` 进行预测

---

## 二、ThinkKT 模块设计需求

### 2.1 模块一：多模态题目编码器（Visual-Language Encoder）

**文件位置**: `thinkkt/encoders/visual_language_encoder.py`

**功能需求：**
1. **图像特征提取**
   - 使用 Qwen2.5-VL 或 InternVL2.5 提取图像特征
   - 输入：题目图片路径（从 `q_imgs` 目录）
   - 输出：图像特征向量 `v_img` (batch, seq_len, d_img)

2. **文本特征提取**（可选）
   - 如果题目有文本描述，提取文本特征
   - 输出：文本特征向量 `v_text` (batch, seq_len, d_text)

3. **知识点分布预测**
   - 轻量分类头，从图像特征预测知识点分布
   - 输出：知识点分布 `k_t` (batch, seq_len, num_c)

4. **特征融合**
   - 融合图像和文本特征：`v_t = [v_img; v_text]` 或门控融合
   - 输出：题目表征 `v_t` (batch, seq_len, d_question)

**接口设计：**
```python
class VisualLanguageEncoder(nn.Module):
    def __init__(self, model_name="qwen2.5-vl", img_dim=1024, num_c=100):
        # 初始化 MLLM 模型
        # 初始化知识点分类头
    
    def forward(self, qids, img_paths, texts=None):
        """
        :param qids: (batch, seq_len) 问题ID
        :param img_paths: dict {qid: img_path} 或直接传入路径列表
        :param texts: (batch, seq_len) 文本列表（可选）
        :return: v_t (batch, seq_len, d_question), k_t (batch, seq_len, num_c)
        """
        pass
    
    def encode_batch(self, img_paths, texts=None):
        """批量编码图像"""
        pass
```

**缓存需求：**
- 预计算所有题目的特征，保存为 `{qid: v_t, k_t}` 字典
- 文件格式：`features/{dataset_name}_question_features.pt`
- 元数据：`features/{dataset_name}_question_features_meta.json`

---

### 2.2 模块二：知识推理链生成器（Knowledge CoT Generator）

**文件位置**: `thinkkt/cot/cot_generator.py`

**功能需求：**
1. **Prompt 模板设计**
   - 结构化模板，包含：
     - 题目考察知识点识别
     - 学生历史掌握情况分析
     - 当前题图像关键信息提取
     - 可能错误原因推理
     - 置信度评估

2. **CoT 生成**
   - 使用 MLLM（Qwen2.5-VL）生成推理链
   - 输入：历史交互 + 当前题目图像
   - 输出：CoT 文本 `r_t` (字符串，长度控制在 80-120 tokens)

3. **CoT 解析与验证**
   - 提取知识点提及
   - 验证格式正确性
   - 过滤空/冗长/格式错误的 CoT

4. **CoT 文本编码**
   - 使用小型文本编码器（MiniLM/DeBERTa-small）编码 CoT
   - 输出：`r_embed` (batch, seq_len, d_cot)

**接口设计：**
```python
class CoTGenerator(nn.Module):
    def __init__(self, mllm_name="qwen2.5-vl", text_encoder_name="minilm"):
        # 初始化 MLLM
        # 初始化文本编码器
        # 加载 prompt 模板
    
    def generate_cot(self, history_qids, history_rs, current_qid, 
                     img_paths, kc_vocab):
        """
        :param history_qids: list[int] 历史问题ID列表
        :param history_rs: list[int] 历史答题结果列表
        :param current_qid: int 当前问题ID
        :param img_paths: dict {qid: img_path}
        :param kc_vocab: dict {kc_id: kc_name} 知识点词表
        :return: cot_text (str), cot_embed (tensor)
        """
        pass
    
    def encode_cot(self, cot_texts):
        """批量编码 CoT 文本"""
        pass
```

**缓存需求：**
- 缓存生成的 CoT：`cot_cache/{dataset_name}_cot_cache.jsonl`
- 格式：`{"qid": int, "history_hash": str, "cot_text": str, "cot_embed": tensor}`

**训练阶段：**
- **阶段1（SFT）**: 使用标注 CoT 或自生成+人工筛选的 CoT 进行监督微调
- **阶段2（RL）**: 使用强化学习优化 CoT 生成质量（见模块四）

---

### 2.3 模块三：知识状态追踪器（Knowledge State Tracker）

**文件位置**: `thinkkt/models/thinkkt_net.py`

**功能需求：**
1. **输入融合**
   - 融合题目特征 `v_t`、答题结果 `a_t`、CoT 嵌入 `r_embed`、知识点分布 `k_t`
   - 输入：`z_i = [v_i; a_i_embed; r_i_embed; k_i]` (batch, seq_len, d_input)

2. **序列建模**
   - 使用 Transformer 或 LSTM 建模时序依赖
   - 输出：知识状态 `h_t` (batch, seq_len, d_knowledge)

3. **预测头**
   - 从知识状态预测答对概率
   - 输出：`ŷ_t = sigmoid(W h_t)` (batch, seq_len)

4. **知识点掌握度输出**（可选）
   - 输出每个知识点的掌握度向量
   - 用于可解释性分析

**接口设计：**
```python
class ThinkKTNet(nn.Module):
    def __init__(self, config):
        # 题目特征维度
        self.d_question = config.get('d_question', 1024)
        # CoT 嵌入维度
        self.d_cot = config.get('d_cot', 384)
        # 知识点数量
        self.num_c = config.get('num_c', 100)
        # 知识状态维度
        self.d_knowledge = config.get('d_knowledge', 512)
        
        # 答题结果嵌入
        self.answer_emb = nn.Embedding(2, self.d_question // 4)
        
        # 特征融合层
        self.fusion_layer = nn.Linear(
            self.d_question + self.d_question // 4 + self.d_cot + self.num_c,
            self.d_knowledge
        )
        
        # 序列建模（Transformer 或 LSTM）
        self.seq_model = nn.TransformerEncoder(...)  # 或 LSTM
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(self.d_knowledge, self.d_knowledge // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.d_knowledge // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, v_t, a_t, r_embed, k_t, mask=None):
        """
        :param v_t: (batch, seq_len, d_question) 题目特征
        :param a_t: (batch, seq_len) 答题结果 (0/1)
        :param r_embed: (batch, seq_len, d_cot) CoT 嵌入
        :param k_t: (batch, seq_len, num_c) 知识点分布
        :param mask: (batch, seq_len) 掩码
        :return: y_pred (batch, seq_len) 答对概率
        """
        # 1. 答题结果嵌入
        a_emb = self.answer_emb(a_t.long())  # (batch, seq_len, d_question//4)
        
        # 2. 特征融合
        z = torch.cat([v_t, a_emb, r_embed, k_t], dim=-1)  # (batch, seq_len, d_input)
        z = self.fusion_layer(z)  # (batch, seq_len, d_knowledge)
        
        # 3. 序列建模
        h_t = self.seq_model(z, src_key_padding_mask=~mask if mask is not None else None)
        
        # 4. 预测
        y_pred = self.predictor(h_t).squeeze(-1)  # (batch, seq_len)
        
        return y_pred
```

---

### 2.4 模块四：强化学习优化器（RL-based Reasoning Reward）

**文件位置**: `thinkkt/rl/cot_rl_trainer.py`

**功能需求：**
1. **奖励设计**
   - **预测准确性奖励** `R_pred`: 使用 CoT 后预测准确性的提升
   - **一致性奖励** `R_cons`: CoT 声称的掌握情况与预测结果的一致性
   - **知识点覆盖奖励** `R_kc`: CoT 提及的知识点与题目标注的重合度
   - **长度惩罚** `R_len`: 控制 CoT 长度在合理范围

2. **RL 训练**
   - 使用 GRPO 或 PPO 优化 CoT 生成器
   - 冻结知识状态追踪器，只优化 CoT 生成器的 LoRA 参数

**接口设计：**
```python
class CoTRLTrainer:
    def __init__(self, cot_generator, kt_model, reward_weights):
        self.cot_generator = cot_generator
        self.kt_model = kt_model  # 冻结
        self.reward_weights = reward_weights
    
    def compute_reward(self, cot_texts, predictions_with_cot, 
                      predictions_without_cot, kc_labels):
        """
        计算奖励
        :return: rewards (batch, seq_len)
        """
        # R_pred: 预测提升
        r_pred = (predictions_with_cot - predictions_without_cot).mean()
        
        # R_cons: 一致性（需要解析 CoT 中的掌握情况）
        r_cons = self._compute_consistency(cot_texts, predictions_with_cot)
        
        # R_kc: 知识点覆盖
        r_kc = self._compute_kc_coverage(cot_texts, kc_labels)
        
        # R_len: 长度惩罚
        r_len = self._compute_length_penalty(cot_texts)
        
        # 加权求和
        reward = (self.reward_weights['pred'] * r_pred +
                  self.reward_weights['cons'] * r_cons +
                  self.reward_weights['kc'] * r_kc +
                  self.reward_weights['len'] * r_len)
        
        return reward
    
    def train_step(self, batch_data):
        """执行一步 RL 训练"""
        pass
```

---

### 2.5 主模型类：ThinkKT

**文件位置**: `thinkkt/models/thinkkt.py`

**接口设计：**
```python
class ThinkKT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = 'thinkkt'
        self.device = self._get_device()
        
        # 初始化各模块
        self.visual_encoder = VisualLanguageEncoder(...)
        self.cot_generator = CoTGenerator(...)
        self.kt_net = ThinkKTNet(config)
        
        # 加载预计算的特征缓存
        self.question_features = self._load_question_features(config)
        self.cot_cache = self._load_cot_cache(config)
    
    def train_one_step(self, data):
        """
        pykt 标准接口
        :param data: 数据字典，包含 qseqs, cseqs, rseqs, shft_qseqs, shft_rseqs, smasks
        :return: (y, loss)
        """
        # 1. 获取题目特征
        qids = data['qseqs']  # (batch, seq_len-1)
        v_t = self._get_question_features(qids)  # (batch, seq_len-1, d_question)
        k_t = self._get_kc_distribution(qids)  # (batch, seq_len-1, num_c)
        
        # 2. 生成或获取 CoT
        r_embed = self._get_cot_embeddings(qids, data['rseqs'])  # (batch, seq_len-1, d_cot)
        
        # 3. 前向传播
        y = self.kt_net(v_t, data['rseqs'], r_embed, k_t, data['masks'])
        
        # 4. 计算损失
        sm = data['smasks'].to(self.device)
        r_shift = data['shft_rseqs'].to(self.device)
        loss = self.get_loss(y, r_shift, sm)
        
        return y, loss
    
    def predict_one_step(self, data):
        """pykt 标准接口"""
        with torch.no_grad():
            y, _ = self.train_one_step(data)
        return y
    
    def get_loss(self, ys, rshft, sm):
        """计算损失"""
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        loss = F.binary_cross_entropy(y_pred.double(), y_true.double())
        return loss
    
    def _get_question_features(self, qids):
        """从缓存获取题目特征"""
        pass
    
    def _get_cot_embeddings(self, qids, rseqs):
        """生成或获取 CoT 嵌入"""
        pass
```

---

## 三、数据流设计

### 3.1 数据预处理流程

1. **题目特征预计算**
   - 脚本：`scripts/precompute_question_features.py`
   - 输入：`q_imgs` 目录下的所有图片
   - 输出：`features/{dataset_name}_question_features.pt`

2. **CoT 预生成（可选）**
   - 脚本：`scripts/precompute_cot.py`
   - 输入：训练数据中的交互序列
   - 输出：`cot_cache/{dataset_name}_cot_cache.jsonl`

3. **数据加载器扩展**
   - 文件：`pykt/datasets/thinkkt_dataloader.py`
   - 继承 `KTDataset`，添加题目特征和 CoT 的加载逻辑

### 3.2 训练时数据流

```
DataLoader
  ↓
KTDataset (扩展)
  ↓
data = {
    'qseqs': ...,
    'cseqs': ...,
    'rseqs': ...,
    'shft_qseqs': ...,
    'shft_rseqs': ...,
    'smasks': ...,
    'question_features': ...,  # 新增：题目特征
    'cot_embeddings': ...       # 新增：CoT 嵌入
}
  ↓
ThinkKT.train_one_step(data)
  ↓
y, loss
```

---

## 四、训练阶段设计

### 4.1 阶段1：监督微调（SFT）

**目标**: 让模型看懂题目、生成基础 CoT

**训练脚本**: `scripts/train_sft.py`

**流程：**
1. 使用预训练的 Qwen2.5-VL 提取题目特征
2. 使用标注的 CoT 或自生成+人工筛选的 CoT 训练 CoT 生成器
3. 训练知识点分类头

**损失函数：**
- CoT 生成损失（交叉熵）
- 知识点分类损失（交叉熵）

### 4.2 阶段2：强化学习优化（RL）

**目标**: 优化 CoT 质量，提升预测准确性

**训练脚本**: `scripts/train_rl.py`

**流程：**
1. 冻结知识状态追踪器
2. 使用 RL 优化 CoT 生成器的 LoRA 参数
3. 奖励函数：预测提升 + 一致性 + 知识点覆盖 + 长度惩罚

### 4.3 阶段3：知识追踪训练（KT）

**目标**: 训练知识状态追踪器

**训练脚本**: `scripts/train_kt.py`（或使用标准 `wandb_train.py`）

**流程：**
1. 固定或半固定 CoT 生成器
2. 训练知识状态追踪器
3. 可选的端到端微调

---

## 五、配置文件设计

### 5.1 模型配置（添加到 `kt_config.json`）

```json
{
    "thinkkt": {
        "learning_rate": 1e-4,
        "d_question": 1024,
        "d_cot": 384,
        "d_knowledge": 512,
        "dropout": 0.1,
        "seq_model_type": "transformer",  // 或 "lstm"
        "num_transformer_layers": 6,
        "num_heads": 8,
        "mllm_name": "qwen2.5-vl-3b",
        "text_encoder_name": "minilm",
        "use_cot": true,
        "use_visual": true
    }
}
```

### 5.2 数据配置扩展

在 `data_config.json` 中添加：
```json
{
    "DBE_KT22": {
        ...
        "q_imgs_dir": "/home3/zhiyu/code-5/CRKT/data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/q_imgs",
        "features_dir": "features",
        "cot_cache_dir": "cot_cache"
    }
}
```

---

## 六、实现优先级

### 优先级1（核心功能）
1. ✅ 多模态题目编码器（Visual-Language Encoder）
2. ✅ 知识状态追踪器（ThinkKTNet）
3. ✅ 主模型类（ThinkKT）实现标准接口
4. ✅ 模型注册（在 `init_model.py` 中添加）

### 优先级2（增强功能）
5. ✅ CoT 生成器（基础版本，不使用 RL）
6. ✅ 数据加载器扩展
7. ✅ 特征预计算脚本

### 优先级3（高级功能）
8. ✅ 强化学习优化器
9. ✅ 三阶段训练脚本
10. ✅ 评估和消融实验脚本

---

## 七、关键挑战与解决方案

### 7.1 挑战1：特征缓存与内存管理

**问题**: 题目特征和 CoT 嵌入可能占用大量内存

**解决方案**:
- 使用 `torch.load(..., map_location='cpu')` 延迟加载
- 使用 `functools.lru_cache` 缓存最近使用的特征
- 考虑使用 HDF5 或 LMDB 存储大型特征

### 7.2 挑战2：CoT 生成速度

**问题**: 实时生成 CoT 可能很慢

**解决方案**:
- 预生成 CoT 并缓存
- 使用批处理加速
- 考虑使用更小的模型（如 Qwen2.5-VL-3B）

### 7.3 挑战3：RL 训练稳定性

**问题**: RL 训练可能不稳定

**解决方案**:
- 使用 GRPO 替代 PPO（更稳定）
- 仔细设计奖励函数，避免极端值
- 使用 reward normalization

---

## 八、测试与验证

### 8.1 单元测试

- 测试各模块的输入输出格式
- 测试特征缓存加载
- 测试 CoT 生成和解析

### 8.2 集成测试

- 测试完整的数据流
- 测试训练循环
- 测试评估流程

### 8.3 消融实验

- w/ vs w/o CoT
- w/ vs w/o 图像特征
- w/ vs w/o RL 优化
- 不同奖励权重的对比

---

## 九、文件结构

```
CRKT/
├── thinkkt/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── thinkkt.py          # 主模型类
│   │   └── thinkkt_net.py      # 知识状态追踪器
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── visual_language_encoder.py
│   │   └── text_encoder.py
│   ├── cot/
│   │   ├── __init__.py
│   │   ├── cot_generator.py
│   │   └── cot_prompts.py      # Prompt 模板
│   └── rl/
│       ├── __init__.py
│       └── cot_rl_trainer.py
├── scripts/
│   ├── precompute_question_features.py
│   ├── precompute_cot.py
│   ├── train_sft.py
│   ├── train_rl.py
│   └── train_kt.py
├── pykt/
│   ├── models/
│   │   ├── init_model.py       # 添加 ThinkKT 注册
│   │   └── our_model/
│   │       └── thinkkt.py      # 或放在这里
│   └── datasets/
│       └── thinkkt_dataloader.py
└── features/                   # 特征缓存目录
    └── {dataset_name}_question_features.pt
```

---

## 十、下一步行动

1. **创建基础文件结构**
   - 创建 `thinkkt/` 目录和子目录
   - 创建基础 `__init__.py` 文件

2. **实现多模态编码器**
   - 实现 `VisualLanguageEncoder`
   - 编写特征预计算脚本

3. **实现知识状态追踪器**
   - 实现 `ThinkKTNet`
   - 测试前向传播

4. **实现主模型类**
   - 实现 `ThinkKT` 类
   - 实现 `train_one_step` 和 `predict_one_step`

5. **集成到 pykt**
   - 在 `init_model.py` 中注册
   - 测试训练流程

6. **实现 CoT 生成器**
   - 设计 prompt 模板
   - 实现基础 CoT 生成

7. **实现 RL 优化器**（可选）
   - 实现奖励函数
   - 实现 RL 训练循环

