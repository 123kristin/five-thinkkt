# GPU并行执行说明

## 当前实现

当前脚本是**串行执行**（使用 `for` 循环），即使分配了不同GPU，同一时间只有一个GPU在工作。

## 原因

- 脚本按顺序执行每个实验：实验1完成后 → 实验2 → 实验3...
- 虽然每个实验被分配到不同GPU，但它们不是同时运行的

## 解决方案

如果要实现真正的并行执行（多个GPU同时工作），需要：

### 方案1：手动启动多个脚本（推荐）

为每个GPU启动一个独立的脚本，分配不同的实验：

thikkt模型基本版本

```bash
# 终端1：运行实验 1,5,9,13,17 (GPU 0)
python run_all_thinkkt_experiments.py --gpu_id "0" --experiment_range "1,2,3,4,5,6"
python wandb_thinkkt_train.py --gpu_id "0" --seq_model_type transformer --num_transformer_layers 2 --save_dir saved_model/baseline_version --dataset_name DBE_KT22
python wandb_thinkkt_train.py --gpu_id "0" --seq_model_type transformer --num_transformer_layers 2 --save_dir saved_model/baseline_version_input --dataset_name DBE_KT22
python wandb_predict.py --save_dir saved_model/baseline_version_input/DBE_KT22_0_0.0001_32_thinkkt_qkcs_1024_384_512_0.1_transformer_2_8_2_False_True_features --gpu_id "0"
python run_all_thinkkt_experiments.py --gpu_id "0" --experiment_range "1"
# 终端2：运行实验 2,6,10,14,18 (GPU 1)  
python run_all_thinkkt_experiments.py --gpu_id "1" --experiment_range "7,8,9,10,11,12"
python run_all_thinkkt_experiments.py --gpu_id "1" --experiment_range "7"
# 终端3：运行实验 3,7,11,15 (GPU 2)
python run_all_thinkkt_experiments.py --gpu_id "2" --experiment_range "13,14,15,16,17,18"
python wandb_thinkkt_train.py --gpu_id "0" --seq_model_type transformer --num_transformer_layers 2 --save_dir saved_model/baseline_version --dataset_name nips_task34
python run_all_thinkkt_experiments.py --gpu_id "2" --experiment_range "13"
```
thinkkt模型CoT版本
```bash
# 终端1：运行实验 1,5,9,13,17 (GPU 0)
python run_all_thinkkt_experiments.py --gpu_id "0" --experiment_range "1,2,3,4,5,6" --use_cot 1
python wandb_thinkkt_train.py --gpu_id "0" --seq_model_type transformer --num_transformer_layers 2 --save_dir saved_model/baseline_version_input --dataset_name DBE_KT22 --use_cot 1
python run_all_thinkkt_experiments.py --gpu_id "0" --experiment_range "1" --use_cot 1
# 终端2：运行实验 2,6,10,14,18 (GPU 1)  
python run_all_thinkkt_experiments.py --gpu_id "1" --experiment_range "7,8,9,10,11,12" --use_cot 1
python run_all_thinkkt_experiments.py --gpu_id "1" --experiment_range "7" --use_cot 1
# 终端3：运行实验 3,7,11,15 (GPU 2)
python run_all_thinkkt_experiments.py --gpu_id "2" --experiment_range "13,14,15,16,17,18" --use_cot 1
python wandb_thinkkt_train.py --gpu_id "0" --seq_model_type transformer --num_transformer_layers 2 --save_dir saved_model/baseline_version --dataset_name nips_task34
python run_all_thinkkt_experiments.py --gpu_id "2" --experiment_range "13" --use_cot 1
```







### 方案2：使用GNU parallel（如果安装了）

```bash
parallel -j 4 python run_all_thinkkt_experiments.py --gpu_id {} ::: 0 1 2 3
```

### 方案3：修改脚本支持并行（需要较大改动）

需要将 `for` 循环改为使用 `multiprocessing.Pool` 或 `concurrent.futures`，这会增加代码复杂度。

## 建议

对于18个实验，串行执行虽然慢一些，但：
- ✅ 代码简单，易于维护
- ✅ 日志清晰，不会混乱
- ✅ 错误处理简单
- ✅ 资源管理简单

如果需要加速，建议使用**方案1**（手动启动多个脚本）。

