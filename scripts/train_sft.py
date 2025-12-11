"""
阶段1：监督微调（SFT）训练脚本
训练 CoT 生成器生成基础推理链
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))
from pykt.models.our_model.cot.cot_generator import CoTGenerator
from pykt.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="SFT 训练 CoT 生成器")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--mllm_name", type=str,
                       default="/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
                       help="MLLM 模型路径")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--save_dir", type=str, default="saved_models/cot_sft", help="保存目录")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CURRENT_GPU_ID'] = args.gpu_id
    set_seed(2025)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化 CoT 生成器
    print(f"[SFT] 正在初始化 CoT 生成器...")
    cot_generator = CoTGenerator(
        mllm_name=args.mllm_name,
        device=device
    )
    
    # 加载训练数据
    # TODO: 实现数据加载逻辑
    # 这里需要加载标注的 CoT 数据或自生成的数据
    
    # 设置优化器（只优化 LoRA 参数或特定层）
    # 如果使用 LoRA，需要先应用 LoRA
    # optimizer = torch.optim.AdamW(cot_generator.parameters(), lr=args.learning_rate)
    
    # 训练循环
    print(f"[SFT] 开始训练...")
    # TODO: 实现训练循环
    # for epoch in range(args.num_epochs):
    #     for batch in train_loader:
    #         # 前向传播
    #         # 计算损失
    #         # 反向传播
    #         pass
    
    print(f"[SFT] 训练完成！")

if __name__ == "__main__":
    main()

