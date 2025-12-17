# CoT生成完整示例

## 场景设置

假设我们正在处理 **XES3G5M** 数据集（中文数据集）中的一个样本。

---

## 一、输入信息

### 1.1 函数调用参数

```python
cot_text, cot_embed = cot_generator.generate_cot(
    history_qids=[15, 23, 8, 42, 5],      # 历史问题ID列表
    history_rs=[1, 0, 1, 1, 0],           # 历史答题结果 (1=答对, 0=答错)
    current_qid=37,                        # 当前问题ID
    img_path="/path/to/q_imgs/37.jpg",    # 当前题目图片路径
    kc_vocab={                             # 知识点词表
        1: "分数运算",
        2: "几何图形",
        3: "代数方程",
        5: "小数运算",
        8: "百分比",
        12: "面积计算",
        15: "函数图像"
    },
    history_kcs=[                          # 历史问题的知识点列表（二维列表）
        [1, 5],      # 问题15涉及知识点1和5
        [2, 12],     # 问题23涉及知识点2和12
        [3],         # 问题8涉及知识点3
        [1, 8],      # 问题42涉及知识点1和8
        [15]         # 问题5涉及知识点15
    ],
    current_kcs=[1, 8]                     # 当前问题37涉及知识点1和8
)
```

### 1.2 实际输入数据说明

**历史交互序列**（最近5条，按时间顺序）：
- 问题15：答对，涉及"分数运算"和"小数运算"
- 问题23：答错，涉及"几何图形"和"面积计算"
- 问题8：答对，涉及"代数方程"
- 问题42：答对，涉及"分数运算"和"百分比"
- 问题5：答错，涉及"函数图像"

**当前题目**：
- 问题ID：37
- 涉及知识点：分数运算、百分比
- 题目图片：`q_imgs/37.jpg`（一张包含分数和百分比计算题目的图片）

---

## 二、构建的Prompt（输入给MLLM）

基于输入信息，系统会自动构建以下提示词（中文版本，因为XES3G5M是中文数据集）：

```text
你是一个知识追踪专家，需要分析学生的学习情况并生成推理链。
请根据学生的历史答题记录和当前题目，生成结构化的推理链。

## 学生历史交互记录：
  问题 15: 答对，涉及知识点：分数运算, 小数运算
  问题 23: 答错，涉及知识点：几何图形, 面积计算
  问题 8: 答对，涉及知识点：代数方程
  问题 42: 答对，涉及知识点：分数运算, 百分比
  问题 5: 答错，涉及知识点：函数图像

## 当前题目：
问题ID: 37
涉及知识点: 分数运算, 百分比

## 请生成推理链（按以下结构）：
1. **题目考察知识点识别**：当前题目主要考察哪些知识点？
2. **学生历史掌握情况**：基于历史记录，学生已掌握/薄弱的知识点有哪些？
3. **图像关键信息**：题目图片中包含哪些关键信息（如几何图形、标注、已知量等）？
4. **可能错误原因**：如果学生答错，可能的原因是什么？如果答对，说明其掌握情况。
5. **预测置信度**：给出0-1之间的置信度分数。

**注意**：
- 如果图片中没有某个标记或信息，请明确说明'缺失'，不要臆测
- 推理链要简洁，控制在80-120个token
- 重点关注知识点级别的分析，而非题目级别
```

**MLLM同时接收**：
- 📷 题目图片（37.jpg）
- 📝 上述文本提示词

---

## 三、MLLM生成的CoT文本（输出示例）

```text
1. **题目考察知识点识别**：当前题目主要考察分数运算和百分比的应用。题目涉及将分数转换为百分比的计算。

2. **学生历史掌握情况**：基于历史记录，学生在分数运算方面表现较好（问题15和42都答对），在百分比方面也有成功经验（问题42答对）。但在几何图形和函数图像方面较薄弱（问题23和5都答错）。

3. **图像关键信息**：题目图片中包含一个分数形式的数值（如3/4），要求学生将其转换为百分比形式。题目还标注了明确的答题要求。

4. **可能错误原因**：由于学生历史在分数运算和百分比相关题目中表现良好，预测学生很可能答对。如果答错，可能的原因是对分数和百分比之间的转换细节不够熟练，或者计算过程中的疏忽。

5. **预测置信度**：置信度：0.75
```

---

## 四、后续处理

### 4.1 CoT文本验证

系统会验证生成的CoT是否符合要求：
- ✅ 包含关键词：'知识点'、'掌握'、'题目'
- ✅ 长度在合理范围内（20-500字符）
- ✅ 符合基本格式要求

### 4.2 CoT文本编码

将CoT文本通过 **Sentence-Transformers** 编码器转换为向量：

```python
# 使用 paraphrase-multilingual-MiniLM-L12-v2 编码器
cot_embed = text_encoder.encode(cot_text, convert_to_tensor=True)
# cot_embed.shape: torch.Size([384])  # d_cot=384
```

**CoT嵌入向量示例**（384维，显示前10维）：
```python
tensor([ 0.0234, -0.0156,  0.0821, -0.0345,  0.0123, 
        -0.0456,  0.0678, -0.0234,  0.0891, -0.0123, ...])
```

### 4.3 缓存存储

生成的CoT会被保存到缓存中（如果启用缓存）：

```json
{
  "cache_key": "a1b2c3d4e5f6...",  // MD5哈希值
  "cot_text": "1. **题目考察知识点识别**：当前题目主要考察分数运算...",
  "cot_embed": [0.0234, -0.0156, 0.0821, ...],  // 384维向量（列表形式）
  "history_qids": [15, 23, 8, 42, 5],
  "history_rs": [1, 0, 1, 1, 0],
  "current_qid": 37
}
```

---

## 五、完整输出

### 5.1 函数返回值

```python
cot_text, cot_embed = cot_generator.generate_cot(...)

# cot_text: str
# 完整的CoT文本（如上所示）

# cot_embed: torch.Tensor
# Shape: (384,)
# 类型: torch.float32
# 设备: cuda:0 或 cpu
```

### 5.2 在ThinkKT中的使用

```python
# 在训练/预测时，CoT嵌入会被使用：
r_embed = torch.stack([cot_embed1, cot_embed2, ..., cot_embed_n])
# r_embed.shape: (batch_size, seq_len, 384)

# 与题目特征、答案特征、知识点分布融合：
z = torch.cat([v_t, a_emb, r_embed, k_t], dim=-1)
# z.shape: (batch_size, seq_len, d_input)
# 其中 d_input = d_question + d_answer + d_cot + num_c
#      = 1024 + 256 + 384 + num_c
```

---

## 六、英文数据集示例（DBE_KT22）

如果是英文数据集（如DBE_KT22），输入和输出会使用英文：

### 6.1 输入参数（相同结构）

```python
cot_text, cot_embed = cot_generator.generate_cot(
    history_qids=[10, 25, 7],
    history_rs=[1, 0, 1],
    current_qid=15,
    img_path="/path/to/q_imgs/15.jpg",
    kc_vocab={
        1: "SQL Queries",
        2: "Database Design",
        3: "Normalization"
    },
    history_kcs=[[1], [2], [3]],
    current_kcs=[1, 2]
)
```

### 6.2 生成的英文Prompt

```text
You are a knowledge tracing expert who needs to analyze student learning situations and generate reasoning chains.
Please generate structured reasoning chains based on the student's historical answer records and the current question.

## Student Historical Interaction Records:
  Question 10: Correct, involving concepts: SQL Queries
  Question 25: Incorrect, involving concepts: Database Design
  Question 7: Correct, involving concepts: Normalization

## Current Question:
Question ID: 15
Involved Concepts: SQL Queries, Database Design

## Please generate a reasoning chain (following this structure):
1. **Knowledge Point Identification**: Which knowledge points does the current question primarily examine?
2. **Student Historical Mastery**: Based on historical records, which knowledge points has the student mastered/weakened?
3. **Image Key Information**: What key information is contained in the question image (e.g., geometric shapes, annotations, known quantities)?
4. **Possible Error Reasons**: If the student answers incorrectly, what might be the reason? If correct, explain their mastery situation.
5. **Prediction Confidence**: Provide a confidence score between 0 and 1.

**Notes**:
- If certain marks or information are not present in the image, clearly state 'missing', do not speculate
- The reasoning chain should be concise, controlled within 80-120 tokens
- Focus on knowledge point-level analysis rather than question-level
```

### 6.3 生成的英文CoT示例

```text
1. **Knowledge Point Identification**: The current question primarily examines SQL Queries and Database Design concepts, focusing on query optimization and relational database structure.

2. **Student Historical Mastery**: Based on historical records, the student shows strong performance in SQL Queries (Question 10 correct) and Normalization (Question 7 correct), but struggles with Database Design (Question 25 incorrect).

3. **Image Key Information**: The question image contains a database schema diagram with multiple tables and relationships, along with a SQL query statement that requires optimization.

4. **Possible Error Reasons**: Given the student's mixed performance, there is moderate confidence. If correct, it indicates improvement in Database Design understanding. If incorrect, it may be due to incomplete grasp of table relationships or query optimization principles.

5. **Prediction Confidence**: Confidence: 0.65
```

---

## 七、生成流程图

```
输入信息
  ├─ history_qids, history_rs, current_qid
  ├─ img_path (题目图片)
  ├─ kc_vocab, history_kcs, current_kcs
  └─ dataset_name (自动识别语言)
        ↓
构建Prompt
  ├─ 根据dataset_name选择语言模板
  ├─ 填充历史交互信息
  ├─ 填充当前题目信息
  └─ 添加推理要求
        ↓
MLLM生成
  ├─ 输入：图片 + 文本Prompt
  ├─ 模型：Qwen2.5-VL-3B-Instruct
  └─ 输出：CoT文本（结构化推理链）
        ↓
验证和编码
  ├─ validate_cot()：验证文本质量
  ├─ text_encoder.encode()：编码为向量
  └─ 维度：384 (d_cot)
        ↓
缓存和返回
  ├─ 保存到缓存（可选）
  └─ 返回：(cot_text, cot_embed)
```

---

## 八、总结

### 输入
- ✅ 历史交互序列（题目ID、答题结果、知识点）
- ✅ 当前题目信息（题目ID、图片路径、知识点）
- ✅ 知识点词表

### 输出
1. **CoT文本**：结构化的推理链，包含：
   - 题目考察知识点识别
   - 学生历史掌握情况
   - 图像关键信息
   - 可能错误原因
   - 预测置信度

2. **CoT嵌入向量**：384维的语义表示
   - 用于后续的知识追踪模型输入
   - 与题目特征、答案特征、知识点分布融合

3. **缓存**（可选）：保存CoT文本和嵌入，避免重复生成

