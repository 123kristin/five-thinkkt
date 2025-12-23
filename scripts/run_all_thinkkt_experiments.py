#!/usr/bin/env python
"""
批量运行ThinkKT模型的所有实验组合
包括：3个数据集 × 2种序列模型类型 × 3种层数 = 18个实验
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, Manager
import time

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))

def run_command(cmd, description, log_file=None):
    """
    运行命令并记录日志
    
    Args:
        cmd: 命令列表
        description: 命令描述
        log_file: 日志文件路径（可选）
    """
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print("=" * 80)
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 80)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}\n")
            f.write(f"{'='*80}\n")
            f.write(f"执行命令: {' '.join(cmd)}\n")
            f.write(f"{'-'*80}\n")
            f.flush()
    
    try:
        # 运行命令，实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(line)
                    f.flush()
        
        process.wait()
        return_code = process.returncode
        
        if return_code != 0:
            print(f"\n❌ 命令执行失败，返回码: {return_code}")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n❌ 命令执行失败，返回码: {return_code}\n")
            return False
        else:
            print(f"\n✅ 命令执行成功")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n✅ 命令执行成功\n")
            return True
            
    except Exception as e:
        print(f"\n❌ 执行命令时出错: {e}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n❌ 执行命令时出错: {e}\n")
        return False


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_dir = os.path.join(curr_dir, "../scripts_training2testing/examples")
    
    parser = argparse.ArgumentParser(description="批量运行ThinkKT实验")
    parser.add_argument("--base_dir", type=str, 
                       default=default_base_dir,
                       help="工作目录（包含wandb_thinkkt_train.py的目录）")
    parser.add_argument("--gpu_id", type=str, default="0", 
                       help="GPU ID（单个）或GPU列表（逗号分隔，如'0,1,2,3'）。如果提供多个GPU，将轮询分配实验")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数")
    parser.add_argument("--use_cot", type=int, default=0, 
                       help="是否使用CoT (0=Baseline, 1=CoT版本)")
    parser.add_argument("--cot_threshold", type=int, default=2,
                        help="CoT生成的稀疏阈值")
    parser.add_argument("--adaptive_strategy", type=str, default="rule", 
                        help="CoT生成策略: 'rule' 或 'learnable'")
    parser.add_argument("--pretrained_model_dir", type=str, default=None,
                        help="预训练模型目录(用于learnable模式跳过Step1)")
    parser.add_argument("--question_rep_type", type=str, default="visual", choices=["visual", "qid"],
                        help="题目表征来源: 'visual' (ThinkKT) 或 'qid' (CRKT)")
                        
    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--skip_training", action="store_true", 
                       help="跳过训练，只运行测试（用于重新测试已训练的模型）")
    parser.add_argument("--skip_testing", action="store_true", 
                       help="跳过测试，只运行训练")
    parser.add_argument("--force", action="store_true",
                       help="强制重新运行所有实验，即使已完成（忽略断点续传）")
    parser.add_argument("--experiment_range", type=str, default=None,
                       help="指定要运行的实验编号（逗号分隔），如'1,5,9,13,17'。如果不指定，则运行所有实验")
    
    args = parser.parse_args()
    
    # 解析GPU列表
    if ',' in args.gpu_id:
        # 多个GPU，解析为列表
        gpu_list = [gpu.strip() for gpu in args.gpu_id.split(',') if gpu.strip()]
        print(f"[GPU分配] 使用多GPU模式: {gpu_list}")
    else:
        # 单个GPU，转换为列表
        gpu_list = [args.gpu_id]
        print(f"[GPU分配] 使用单GPU模式: {gpu_list}")
    
    # 实验配置（注意：配置文件中使用的是'nips_task34'，但显示名称可以用'NIPS_task34'）
    datasets = ["DBE_KT22", "XES3G5M", "nips_task34"]
    seq_model_types = ["lstm", "transformer"]
    num_layers_options = [1, 2, 3]
    
    # 切换到工作目录
    original_dir = os.getcwd()
    os.chdir(args.base_dir)
    
    # 创建日志目录
    log_dir = "experiment_input_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 总日志文件（使用绝对路径，避免切换目录后找不到）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_log = os.path.join(os.getcwd(), log_dir, f"all_experiments_{timestamp}.log")
    
    # 统计信息
    total_experiments = 0
    completed_experiments = 0
    skipped_experiments = []  # 跳过的已完成实验
    failed_experiments = []
    
    datasets = ["DBE_KT22", "XES3G5M", "nips_task34"]
    question_rep_types = ["qid", "visual"]
    num_lstm_layers_options = [1, 2, 3]
    
    # 生成所有实验组合 (3x2x3 = 18个)
    experiments = []
    for dataset in datasets:
        for q_rep in question_rep_types:
            for num_layers in num_lstm_layers_options:
                experiments.append({
                    'dataset': dataset,
                    'question_rep_type': q_rep,
                    'seq_model_type': 'lstm',
                    'num_lstm_layers': num_layers,
                    'num_transformer_layers': None
                })
    
    total_experiments = len(experiments)
    
    # 如果指定了experiment_range，筛选要运行的实验
    experiment_indices_map = {}  # 映射：当前索引 -> 原始实验编号（用于GPU分配）
    if args.experiment_range:
        try:
            # 解析实验编号（从1开始）
            exp_indices = [int(x.strip()) for x in args.experiment_range.split(',')]
            # 转换为0-based索引，并筛选有效的实验
            valid_indices = [idx - 1 for idx in exp_indices if 1 <= idx <= total_experiments]
            if valid_indices:
                # 保存映射关系：当前索引 -> 原始实验编号
                filtered_experiments = []
                for new_idx, orig_idx in enumerate(valid_indices):
                    filtered_experiments.append(experiments[orig_idx])
                    experiment_indices_map[new_idx] = orig_idx + 1  # 原始编号（从1开始）
                experiments = filtered_experiments
                print(f"[实验筛选] 根据 --experiment_range={args.experiment_range}，筛选出 {len(experiments)} 个实验")
                print(f"[实验筛选] 原始实验编号: {[i+1 for i in valid_indices]}")
            else:
                print(f"[警告] 没有有效的实验编号，将运行所有实验")
        except ValueError as e:
            print(f"[警告] 解析 --experiment_range 失败: {e}，将运行所有实验")
    else:
        # 没有指定范围，所有实验都按原始编号
        for i in range(len(experiments)):
            experiment_indices_map[i] = i + 1
    
    total_experiments = len(experiments)
    
    print("=" * 80)
    print("ThinkKT 批量实验脚本")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    if args.experiment_range:
        print(f"实验范围: {args.experiment_range}")
    print(f"数据集: {datasets}")
    print(f"序列模型类型: {seq_model_types}")
    print(f"层数选项: {num_layers_options}")
    print(f"使用CoT: {args.use_cot}")
    print(f"GPU列表: {gpu_list}")
    print(f"Fold: {args.fold}")
    print(f"强制重新运行: {args.force}")
    print(f"断点续传: {'禁用' if args.force else '启用'}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"日志文件: {master_log}")
    print("=" * 80)
    
    with open(master_log, 'w', encoding='utf-8') as f:
        f.write(f"ThinkKT 批量实验日志\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write("=" * 80 + "\n")
    
    # 运行每个实验
    # 注意：idx 是当前循环中的索引（从1开始），用于GPU轮询分配
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"实验 {idx}/{total_experiments}")
        print(f"{'='*80}")
        
        # 轮询分配GPU（基于原始实验编号，而不是当前循环索引）
        original_exp_idx = experiment_indices_map.get(idx - 1, idx)  # 获取原始实验编号
        assigned_gpu = gpu_list[(original_exp_idx - 1) % len(gpu_list)]
        print(f"分配GPU: cuda:{assigned_gpu} (当前循环索引: {idx}, 原始实验编号: {original_exp_idx})")
        
        # 构建保存目录名称
        if args.use_cot:
             version_name = "cot_version_input"  # CoT 版本 (Group 3)
        else:
             # Baseline 版本 (Group 1 & 2)
             if exp['question_rep_type'] == 'qid':
                 version_name = "crkt_baseline"   # Group 1: CRKT 复刻
             else:
                 version_name = "visual_baseline" # Group 2: Visual 基线

        base_save_dir = f"saved_model/{version_name}"
        
        exp_name = f"{exp['dataset']}_{exp['question_rep_type']}_{exp['seq_model_type']}_L{exp['num_lstm_layers']}"
        
        # save_dir会被训练脚本自动生成完整路径，这里只提供基础目录
        save_dir = base_save_dir
        
        print(f"数据集: {exp['dataset']}")
        print(f"表征类型: {exp['question_rep_type']}")
        print(f"序列模型: {exp['seq_model_type']}")
        print(f"层数: {exp['num_lstm_layers']}")
        print(f"保存目录: {save_dir}")
        
        # 实验日志
        exp_log = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
        
        # 断点续传：检查实验是否已完成
        base_save_dir_full = os.path.join(args.base_dir, f"saved_model/{version_name}")
        existing_model_dir = None
        is_completed = False
        
        if not args.force:
            # 查找已存在的模型目录
            if os.path.exists(base_save_dir_full):
                # 构建匹配关键词（数据集名称 + 序列模型类型）
                match_keywords = [exp['dataset'], exp['seq_model_type']]
                
                # 查找匹配的模型目录
                for item in os.listdir(base_save_dir_full):
                    item_path = os.path.join(base_save_dir_full, item)
                    if not os.path.isdir(item_path):
                        continue
                    
                    # 检查目录名是否包含所有关键词
                    if all(keyword in item for keyword in match_keywords):
                        # 检查是否包含模型文件和测试结果（判断是否完成）
                        model_file = None
                        test_file = None
                        config_file = None
                        
                        for f in os.listdir(item_path):
                            if f.endswith("_model.ckpt"):
                                model_file = os.path.join(item_path, f)
                            elif f == "config.json":
                                config_file = os.path.join(item_path, f)
                            # 检查测试结果文件（必须存在才认为实验完成）
                            if f.endswith("_test_predictions.txt") or (f.startswith("predicting") and f.endswith(".log")):
                                test_file = os.path.join(item_path, f)
                        
                        # 实验完成的条件：必须有模型文件、配置文件、和测试结果文件
                        if (model_file and os.path.exists(model_file) and 
                            config_file and os.path.exists(config_file) and
                            test_file and os.path.exists(test_file)):
                            # 进一步验证配置是否匹配（通过读取config.json）
                            try:
                                import json
                                with open(config_file, 'r') as f:
                                    saved_config = json.load(f)
                                    saved_params = saved_config.get('params', {})
                                    # 检查关键参数是否匹配（包括层数）
                                    saved_num_lstm_layers = saved_params.get('num_lstm_layers')
                                    saved_num_transformer_layers = saved_params.get('num_transformer_layers')
                                    exp_num_layers = exp['num_lstm_layers'] or exp['num_transformer_layers']
                                    
                                    # 匹配条件：数据集名称、序列模型类型、层数都要匹配
                                    if (saved_params.get('dataset_name') == exp['dataset'] and
                                        saved_params.get('seq_model_type') == exp['seq_model_type'] and
                                        saved_params.get('question_rep_type', 'visual') == exp['question_rep_type']):
                                        # 检查层数是否匹配
                                        saved_num_layers = saved_num_lstm_layers or saved_num_transformer_layers
                                        if saved_num_layers == exp_num_layers:
                                            existing_model_dir = item_path
                                            is_completed = True
                                            break
                            except Exception as e:
                                # 如果读取配置失败，不认为已完成（避免误判）
                                pass
        
        if is_completed and not args.force:
            print(f"⏭️  实验已完成，跳过: {exp_name}")
            print(f"   模型目录: {existing_model_dir}")
            skipped_experiments.append(exp_name)
            completed_experiments += 1
            # 记录到总日志
            with open(master_log, 'a', encoding='utf-8') as f:
                f.write(f"\n实验 {idx}/{total_experiments}: {exp_name}\n")
                f.write(f"状态: 已跳过（已完成）\n")
                f.write(f"模型目录: {existing_model_dir}\n")
                f.write(f"{'-'*80}\n")
            continue
        
        success = True
        actual_model_dir = None  # 记录实际模型保存路径
        train_start_time = None  # 记录训练开始时间
        
        # 1. 训练 (Phase 1: Base Model)
        run_phase1 = not args.skip_training
        actual_model_dir = None
        
        # 智能跳过逻辑: 如果是 Learnable 模式，且能找到已存在的基线模型，则跳过 Phase 1
        if args.adaptive_strategy == 'learnable':
            # 搜索最近的一个可用模型目录
            if os.path.exists(save_dir):
                subdirs = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
                # 按修改时间排序，找最新的
                subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                for d in subdirs:
                    if os.path.exists(os.path.join(d, "config.json")):  # 检查配置是否存在，这是最基本的
                        
                        # 额外检查: 确保这个模型不是 RL 训练出来的 (rl_model.pt) 而是 Base 模型
                        # 但通常 wandb_train 生成的目录里会有 config.json
                        print(f"🔄 [Auto-Skip] 发现已有基线模型，跳过 Phase 1，直接进入 RL 训练: {d}")
                        actual_model_dir = d
                        run_phase1 = False
                        break
        
        # 如果用户手动指定了预训练模型 (覆盖自动搜索)
        if args.pretrained_model_dir:
             print(f"🔄 [Manual-Skip] 使用指定基线模型: {args.pretrained_model_dir}")
             actual_model_dir = args.pretrained_model_dir
             run_phase1 = False

        if run_phase1:
            train_start_time = datetime.now()  # 记录训练开始时间
            train_cmd = [
                "python", "wandb_thinkkt_train.py",
                "--dataset_name", exp['dataset'],
                "--fold", str(args.fold),
                "--seq_model_type", exp['seq_model_type'],
                "--use_cot", str(args.use_cot),
                "--use_visual", "1",
                "--save_dir", save_dir,
                "--num_epochs", str(args.num_epochs),
                "--batch_size", str(args.batch_size),
                "--gpu_id", assigned_gpu,  # 使用轮询分配的GPU
                "--cot_threshold", str(args.cot_threshold),
                "--adaptive_strategy", args.adaptive_strategy,
                "--question_rep_type", exp['question_rep_type'] # 使用实验特定的表征类型
            ]
            
            if exp['num_transformer_layers'] is not None:
                train_cmd.extend(["--num_transformer_layers", str(exp['num_transformer_layers'])])
            
            if exp['num_lstm_layers'] is not None:
                train_cmd.extend(["--num_lstm_layers", str(exp['num_lstm_layers'])])
            
            success = run_command(
                train_cmd,
                f"训练实验: {exp_name}",
                log_file=exp_log
            )
            
            if not success:
                print(f"❌ 训练失败: {exp_name}")
                failed_experiments.append(exp_name)
                continue
            
            # 训练完成后，从日志中提取实际保存路径
            # 我们需要解析日志文件来找到 "最佳模型保存在: ..." 的行，或者直接根据规则推断
            # 为了简单，我们让 wandb_thinkkt_train.py 最后打印一行特殊标记，例如 [RESULT_DIR]: /path/to/dir
            # 或者我们直接根据 save_dir 和 exp_name 猜测
            
            # 这里尝试简单推断: save_dir/cot_version_input/dataset_model_layer
            # 但 wandb_train.py 会添加 uuid, 所以最好是从日志读
            if args.adaptive_strategy == 'learnable':
                # 读取日志寻找路径
                if os.path.exists(exp_log):
                    with open(exp_log, 'r') as f:
                        for line in f:
                            if "模型目录:" in line: # wandb_train.py 需要打印这个
                                actual_model_dir = line.split(":")[-1].strip()
                                break
                                
                if not actual_model_dir:
                    print(f"⚠️ 无法找到预训练模型路径，跳过RL训练")
                else:
                    print(f"🔄 检测到 learnable 策略，开始 RL 训练...")
                    print(f"   基础模型路径: {actual_model_dir}")
                    
                    rl_log = os.path.join(save_dir, f"rl_train_{exp_name}.log")
                    rl_cmd = [
                        "python", "scripts/train_rl.py",
                        "--dataset_name", exp['dataset'],
                        "--kt_model_path", actual_model_dir,
                        "--fold", str(args.fold),
                        "--gpu_id", assigned_gpu,
                        "--lambda_cost", "0.1" # 默认值
                    ]
                    
                    success_rl = run_command(rl_cmd, f"RL训练: {exp_name}", log_file=rl_log)
                    if success_rl:
                        print(f"✅ RL训练完成")
                    else:
                        print(f"❌ RL训练失败")
            
            if os.path.exists(exp_log):
                with open(exp_log, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                    # 从后往前查找（路径通常在最后）
                    for line in reversed(log_lines):
                        # 查找包含模型保存路径的行
                        if 'saved_model' in line:
                            # 尝试提取路径
                            for word in line.split():
                                if 'saved_model' in word and exp['dataset'] in word:
                                    potential = word.strip("'\"(),[]\\n:")
                                    # 构建完整路径
                                    if not os.path.isabs(potential):
                                        potential = os.path.join(args.base_dir, potential)
                                    if os.path.exists(potential) and os.path.isdir(potential):
                                        actual_model_dir = potential
                                        break
                            if actual_model_dir:
                                break
        
        # 2. 测试（需要找到实际保存的模型路径）
        if not args.skip_testing and success:
            # 等待一下，确保模型文件已保存
            import time
            time.sleep(3)
            
            # 查找模型保存路径
            # 实际路径格式：saved_model/{version_name}/{dataset}_{fold}_{lr}_{batch}_{model}_{emb}_{...}
            base_save_dir_full = os.path.join(args.base_dir, f"saved_model/{version_name}")
            model_save_dir = None
            
            # 方法1: 使用训练后记录的路径（如果已提取）
            if actual_model_dir and os.path.exists(actual_model_dir):
                model_save_dir = actual_model_dir
            
            # 方法0: 如果已有已完成的模型目录（断点续传的情况），优先使用
            if model_save_dir is None and existing_model_dir and os.path.exists(existing_model_dir):
                model_save_dir = existing_model_dir
            
            # 方法2: 在base_save_dir中查找最近创建的、包含数据集名称的目录
            if model_save_dir is None and os.path.exists(base_save_dir_full):
                matching_dirs = []
                for item in os.listdir(base_save_dir_full):
                    item_path = os.path.join(base_save_dir_full, item)
                    if os.path.isdir(item_path):
                        # 检查是否匹配：包含数据集名称
                        if exp['dataset'] in item:
                            # 检查创建时间（应该在训练开始之后）
                            mtime = os.path.getmtime(item_path)
                            if train_start_time is None or mtime >= train_start_time.timestamp() - 60:  # 允许1分钟的误差
                                matching_dirs.append((item_path, mtime))
                
                if matching_dirs:
                    # 使用最新的目录
                    matching_dirs.sort(key=lambda x: x[1], reverse=True)
                    model_save_dir = matching_dirs[0][0]
            
            # 方法3: 从训练日志中提取路径
            if model_save_dir is None and os.path.exists(exp_log):
                with open(exp_log, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                    for line in reversed(log_lines):  # 从后往前查找（通常路径在最后）
                        if 'saved_model' in line or 'save_dir' in line.lower():
                            # 尝试提取路径
                            for word in line.split():
                                if 'saved_model' in word:
                                    potential = word.strip("'\"(),[]\\n")
                                    if not os.path.isabs(potential):
                                        potential = os.path.join(args.base_dir, potential)
                                    if os.path.exists(potential) and os.path.isdir(potential):
                                        model_save_dir = potential
                                        break
                            if model_save_dir:
                                break
            
            if model_save_dir and os.path.exists(model_save_dir):
                # 检查模型文件是否存在
                model_file = None
                for f in os.listdir(model_save_dir):
                    if f.endswith("_model.ckpt"):
                        model_file = os.path.join(model_save_dir, f)
                        break
                
                if model_file and os.path.exists(model_file):
                    test_cmd = [
                        "python", "wandb_predict.py",
                        "--save_dir", model_save_dir,
                        "--bz", str(args.batch_size),
                        "--gpu_id", assigned_gpu  # 使用与训练相同的GPU
                    ]
                    
                    test_success = run_command(
                        test_cmd,
                        f"测试实验: {exp_name}",
                        log_file=exp_log
                    )
                    
                    if not test_success:
                        print(f"⚠️ 测试失败: {exp_name}")
                else:
                    print(f"⚠️ 模型文件不存在，跳过测试")
                    print(f"   查找目录: {model_save_dir}")
            else:
                print(f"⚠️ 无法找到模型保存目录，跳过测试")
                print(f"   尝试查找: {base_save_dir_full}")
        
        if success:
            completed_experiments += 1
            print(f"✅ 实验完成: {exp_name}")
        else:
            print(f"❌ 实验失败: {exp_name}")
        
        # 记录到总日志
        with open(master_log, 'a', encoding='utf-8') as f:
            f.write(f"\n实验 {idx}/{total_experiments}: {exp_name}\n")
            f.write(f"分配GPU: cuda:{assigned_gpu}\n")
            f.write(f"状态: {'成功' if success else '失败'}\n")
            f.write(f"{'-'*80}\n")
    
    # 恢复原始目录
    os.chdir(original_dir)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    print(f"完成实验: {completed_experiments}")
    print(f"跳过实验: {len(skipped_experiments)}")
    print(f"失败实验: {len(failed_experiments)}")
    if skipped_experiments:
        print(f"\n跳过的实验（已完成）:")
        for exp in skipped_experiments:
            print(f"  - {exp}")
    if failed_experiments:
        print(f"\n失败的实验:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    print(f"\n总日志文件: {master_log}")
    print("=" * 80)
    
    # 保存总结到日志
    with open(master_log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"实验总结\n")
        f.write(f"{'='*80}\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"完成实验: {completed_experiments}\n")
        f.write(f"跳过实验: {len(skipped_experiments)}\n")
        f.write(f"失败实验: {len(failed_experiments)}\n")
        if skipped_experiments:
            f.write(f"跳过的实验（已完成）:\n")
            for exp in skipped_experiments:
                f.write(f"  - {exp}\n")
        if failed_experiments:
            f.write(f"失败的实验:\n")
            for exp in failed_experiments:
                f.write(f"  - {exp}\n")


if __name__ == "__main__":
    main()

