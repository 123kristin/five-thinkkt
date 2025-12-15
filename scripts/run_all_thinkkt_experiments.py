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
    parser = argparse.ArgumentParser(description="批量运行ThinkKT实验")
    parser.add_argument("--base_dir", type=str, 
                       default="/home3/zhiyu/code-5/CRKT/five-thinkkt/scripts_training2testing/examples",
                       help="工作目录（包含wandb_thinkkt_train.py的目录）")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数")
    parser.add_argument("--use_cot", type=int, default=0, 
                       help="是否使用CoT (0=Baseline, 1=CoT版本)")
    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--skip_training", action="store_true", 
                       help="跳过训练，只运行测试（用于重新测试已训练的模型）")
    parser.add_argument("--skip_testing", action="store_true", 
                       help="跳过测试，只运行训练")
    
    args = parser.parse_args()
    
    # 实验配置
    datasets = ["DBE_KT22", "XES3G5M", "NIPS_task34"]
    seq_model_types = ["lstm", "transformer"]
    num_layers_options = [1, 2, 3]
    
    # 切换到工作目录
    original_dir = os.getcwd()
    os.chdir(args.base_dir)
    
    # 创建日志目录
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 总日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_log = os.path.join(log_dir, f"all_experiments_{timestamp}.log")
    
    # 统计信息
    total_experiments = 0
    completed_experiments = 0
    failed_experiments = []
    
    # 生成所有实验组合
    experiments = []
    for dataset in datasets:
        for seq_model_type in seq_model_types:
            if seq_model_type == "transformer":
                for num_layers in num_layers_options:
                    experiments.append({
                        'dataset': dataset,
                        'seq_model_type': seq_model_type,
                        'num_transformer_layers': num_layers,
                        'num_lstm_layers': None
                    })
            else:  # lstm
                for num_layers in num_layers_options:
                    experiments.append({
                        'dataset': dataset,
                        'seq_model_type': seq_model_type,
                        'num_transformer_layers': None,
                        'num_lstm_layers': num_layers
                    })
    
    total_experiments = len(experiments)
    
    print("=" * 80)
    print("ThinkKT 批量实验脚本")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    print(f"数据集: {datasets}")
    print(f"序列模型类型: {seq_model_types}")
    print(f"层数选项: {num_layers_options}")
    print(f"使用CoT: {args.use_cot}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Fold: {args.fold}")
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
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"实验 {idx}/{total_experiments}")
        print(f"{'='*80}")
        
        # 构建保存目录名称（实际路径会在训练时自动生成，这里只是基础目录）
        version_name = "cot_version" if args.use_cot else "baseline_version"
        base_save_dir = f"saved_model/{version_name}"
        
        exp_name = f"{exp['dataset']}_{exp['seq_model_type']}_layers{exp['num_transformer_layers'] or exp['num_lstm_layers']}"
        
        # save_dir会被训练脚本自动生成完整路径，这里只提供基础目录
        save_dir = base_save_dir
        
        print(f"数据集: {exp['dataset']}")
        print(f"序列模型: {exp['seq_model_type']}")
        print(f"层数: {exp['num_transformer_layers'] or exp['num_lstm_layers']}")
        print(f"保存目录: {save_dir}")
        
        # 实验日志
        exp_log = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
        
        success = True
        actual_model_dir = None  # 记录实际模型保存路径
        train_start_time = None  # 记录训练开始时间
        
        # 1. 训练
        if not args.skip_training:
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
                "--gpu_id", args.gpu_id
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
                        "--gpu_id", args.gpu_id
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
    print(f"失败实验: {len(failed_experiments)}")
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
        f.write(f"失败实验: {len(failed_experiments)}\n")
        if failed_experiments:
            f.write(f"失败的实验:\n")
            for exp in failed_experiments:
                f.write(f"  - {exp}\n")


if __name__ == "__main__":
    main()

