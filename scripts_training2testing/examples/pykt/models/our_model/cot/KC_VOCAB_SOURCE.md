# kc_vocab 来源说明

## 概述

`kc_vocab` 是知识点词表，用于将知识点ID映射到知识点名称。它在CoT生成时用于构建更可读的提示词。

**格式**：`{kc_id: kc_name}`，例如：
```python
{
    1: "分数运算",
    2: "几何图形",
    8: "百分比",
    12: "面积计算"
}
```

---

## 数据来源

`kc_vocab` 从各个数据集的原始文件中加载，不同数据集的文件位置和格式不同：

### 1. DBE_KT22 数据集

**文件路径**：`{dpath}/KCs.csv`

**文件格式**：CSV文件，包含以下列：
- `id`: 知识点ID（整数）
- `name`: 知识点名称（字符串）
- `description`: 知识点描述（字符串，可选）

**示例**：
```csv
"id","name","description"
56,Data Model,Data Models are fundamental entities...
12,Subset,"If A and B are two sets..."
24,CREATE TABLE,The CREATE TABLE statement is used...
33,Join,When we want to retrieve data from more than one relation...
```

**加载逻辑**：
```python
df_kcs = pd.read_csv(os.path.join(dpath, "KCs.csv"))
for _, row in df_kcs.iterrows():
    kc_id = int(row['id'])
    kc_name = str(row['name']).strip()
    kc_vocab[kc_id] = kc_name
```

---

### 2. XES3G5M 数据集

**文件路径**：`{dpath}/metadata/kc_routes_map.json` 或 `{dpath}/kc_routes_map.json`

**文件格式**：JSON文件，键为知识点ID（字符串），值为知识点名称（字符串）

**示例**：
```json
{
    "0": "乘法原理里的其他类型",
    "1": "加乘原理综合",
    "2": "分数运算",
    "3": "几何图形",
    ...
}
```

**加载逻辑**：
```python
with open(kc_file, 'r', encoding='utf-8') as f:
    kc_map = json.load(f)
for kc_id_str, kc_name in kc_map.items():
    kc_id = int(kc_id_str)
    kc_vocab[kc_id] = str(kc_name).strip()
```

---

### 3. NIPS_task34 / Eedi 数据集

**注意**：NIPS_task34 和 Eedi 是同一个数据集（Eedi 是源数据集，NIPS_task34 是其在 NIPS 2020 比赛中的名称）。

**文件路径**：`{dpath}/metadata/subject_metadata.csv`

**文件格式**：CSV文件，包含以下列：
- `SubjectId`: 主题ID（整数）
- `Name`: 主题名称（字符串）
- `ParentId`: 父主题ID（用于层次结构，可为NULL）
- `Level`: 层次级别（整数）

**示例**：
```csv
SubjectId,Name,ParentId,Level
3,Maths,NULL,0
32,Number,3,1
33,BIDMAS,144,3
34,Upper and Lower Bounds,141,3
35,Calculator Use,32,2
```

**加载逻辑**：
```python
df_subjects = pd.read_csv(os.path.join(dpath, "metadata", "subject_metadata.csv"))
for _, row in df_subjects.iterrows():
    subject_id = int(row['SubjectId'])
    subject_name = str(row['Name']).strip()
    if pd.notna(subject_id) and pd.notna(subject_name):
        kc_vocab[subject_id] = subject_name
```

**说明**：
- 该数据集使用 **Subject（主题）** 作为知识点的概念
- 训练数据文件（`train_task_3_4.csv`）中没有直接的知识点列，但可以通过 `question_metadata_task_3_4.csv` 中的 `SubjectId` 字段关联到主题
- `subject_metadata.csv` 包含了所有主题的ID和名称映射，可以用作知识点词表

---

## 代码位置

### 加载函数

**文件**：`pykt/models/our_model/thinkkt.py`

**函数**：`load_kc_vocab(dataset_name: str, data_config: dict) -> Dict[int, str]`

**调用位置**：`ThinkKT.__init__()` 方法中

```python
# 在 ThinkKT.__init__() 中
self.kc_vocab = load_kc_vocab(self.dataset_name, data_config)
```

---

## 使用场景

### 1. CoT提示词构建

在构建CoT提示词时，使用 `kc_vocab` 将知识点ID转换为可读的名称：

```python
# 在 cot_prompts.py 中
kc_names = [kc_vocab.get(kc_id, f"知识点{kc_id}") 
            for kc_id in current_kcs]
# 输出: ["分数运算", "百分比"] 而不是 [1, 8]
```

### 2. 默认CoT生成

当MLLM生成失败时，使用知识点名称生成默认CoT：

```python
# 在 cot_generator.py 中
if current_kcs:
    kc_names = [kc_vocab.get(kc, f"知识点{kc}") 
                for kc in current_kcs[:5]]
    kc_text = f"当前题目考察知识点：{', '.join(kc_names)}。"
```

---

## 数据流程

```
数据集原始文件
  ├─ DBE_KT22: KCs.csv
  ├─ XES3G5M: metadata/kc_routes_map.json
  └─ NIPS_task34/Eedi: metadata/subject_metadata.csv
        ↓
load_kc_vocab() 函数
  ├─ 读取文件
  ├─ 解析格式
  └─ 构建字典 {kc_id: kc_name}
        ↓
ThinkKT.__init__()
  └─ self.kc_vocab = load_kc_vocab(...)
        ↓
CoT生成时使用
  ├─ build_cot_prompt() 中使用
  └─ generate_cot() 中使用
```

---

## 注意事项

1. **空词表处理**：如果 `kc_vocab` 为空（加载失败或数据集没有知识点文件），CoT生成时会使用默认格式 `"知识点{kc_id}"` 或 `"Knowledge Point {kc_id}"`。

2. **ID映射**：确保知识点ID与数据集中使用的ID一致。某些数据集可能需要对ID进行映射（如 `qid_ori2new`）。

3. **编码问题**：读取文件时注意编码格式，XES3G5M使用UTF-8编码。

4. **路径问题**：`dpath` 必须正确指向数据集根目录，否则无法找到知识点文件。

---

## 验证方法

可以通过以下方式验证 `kc_vocab` 是否正确加载：

```python
# 在模型初始化后
print(f"知识点词表大小: {len(model.kc_vocab)}")
print(f"示例知识点: {list(model.kc_vocab.items())[:5]}")
```

预期输出：
```
[ThinkKT] 从 .../KCs.csv 加载了 98 个知识点
知识点词表大小: 98
示例知识点: [(56, 'Data Model'), (12, 'Subset'), (24, 'CREATE TABLE'), ...]
```

---

## 扩展新数据集

如果要为新的数据集添加 `kc_vocab` 加载逻辑，可以在 `load_kc_vocab` 函数中添加新的 `elif` 分支：

```python
elif "NewDataset" in dataset_name:
    # 1. 确定知识点文件路径
    kc_file = os.path.join(dpath, "path/to/kc_file")
    
    # 2. 检查文件是否存在
    if os.path.exists(kc_file):
        # 3. 读取并解析文件
        # 4. 构建 kc_vocab 字典 {kc_id: kc_name}
        pass
    else:
        print(f"[ThinkKT] 警告: 找不到文件 {kc_file}")
```

**注意事项**：
- 确保 `kc_vocab` 的格式为 `{int: str}`，即键为整数类型的知识点ID，值为字符串类型的知识点名称
- 如果数据集没有知识点文件，`kc_vocab` 可以为空字典，CoT生成时会使用默认格式 `"知识点{kc_id}"`

