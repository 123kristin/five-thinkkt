"""
阶段2：强化学习优化（RL）训练脚本
使用 RL 优化 CoT 生成质量
"""
import os
import sys
import argparse
import json
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))
from pykt.models.our_model.cot.cot_generator import CoTGenerator
from pykt.models.our_model.rl.cot_rl_trainer import CoTRLTrainer
from pykt.models import init_model, load_model
from pykt.datasets import init_dataset4train
from pykt.models.our_model.visual_language_encoder import build_img_path_dict
from pykt.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="RL 训练优化 CoT 生成器")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数")
    parser.add_argument("--kt_model_path", type=str, required=True, help="知识追踪模型路径（已训练）")
    parser.add_argument("--mllm_name", type=str,
                       default="/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
                       help="MLLM 模型路径")
    parser.add_argument("--d_cot", type=int, default=384, help="CoT嵌入维度")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--save_dir", type=str, default="saved_models/cot_rl", help="保存目录")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--data_config_path", type=str, 
                       default="/home3/zhiyu/code-4/kt_analysis_generation/my_configs/data_config.json",
                       help="数据配置文件路径")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CURRENT_GPU_ID'] = args.gpu_id
    set_seed(2025)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载数据配置
    with open(args.data_config_path) as fin:
        data_configs = json.load(fin)
        data_config = data_configs[args.dataset_name]
    
    # 加载KT模型配置
    kt_config_path = os.path.join(args.kt_model_path, "config.json")
    if not os.path.exists(kt_config_path):
        raise FileNotFoundError(f"找不到KT模型配置文件: {kt_config_path}")
    
    with open(kt_config_path) as fin:
        kt_config = json.load(fin)
        model_config = copy.deepcopy(kt_config["model_config"])
        # 移除不需要的参数
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
    
    # 加载已训练的 KT 模型（冻结）
    print(f"[RL] 正在加载知识追踪模型: {args.kt_model_path}")
    emb_type = kt_config["params"]["emb_type"]
    kt_model = load_model("thinkkt", model_config, data_config, emb_type, args.kt_model_path)
    kt_model.eval()  # 设置为评估模式
    print(f"[RL] KT模型加载完成")
    
    # 初始化 CoT 生成器
    print(f"[RL] 正在初始化 CoT 生成器...")
    cot_generator = CoTGenerator(
        mllm_name=args.mllm_name,
        d_cot=args.d_cot,
        cache_dir=os.path.join(args.save_dir, "cot_cache"),
        device=device,
        use_cache=False  # RL训练时禁用缓存
    )
    print(f"[RL] CoT生成器初始化完成")
    
    # 初始化 RL 训练器
    print(f"[RL] 正在初始化 RL 训练器...")
    rl_trainer = CoTRLTrainer(
        cot_generator=cot_generator,
        kt_model=kt_model,
        reward_weights={
            'pred': 1.0,
            'cons': 0.5,
            'kc': 0.3,
            'len': 0.1
        }
    )
    print(f"[RL] RL训练器初始化完成")
    
    # 加载训练数据
    print(f"[RL] 正在加载训练数据...")
    train_loader, valid_loader, *_ = init_dataset4train(
        args.dataset_name, "thinkkt", data_config, args.fold, args.batch_size
    )
    print(f"[RL] 数据加载完成")
    
    # 构建图片路径映射和知识点词表
    img_path_dict = build_img_path_dict(args.dataset_name, data_config)
    kc_vocab = {}  # TODO: 从数据中加载知识点词表
    
    # 设置优化器（只优化 CoT 生成器的参数）
    optimizer = torch.optim.AdamW(
        cot_generator.parameters(), 
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    print(f"[RL] 开始训练...")
    best_reward = -float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        cot_generator.train()
        epoch_rewards = []
        epoch_losses = []
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # 准备批次数据
            batch_data = {
                'qseqs': data['qseqs'],
                'rseqs': data['rseqs'],
                'shft_rseqs': data['shft_rseqs'],
                'cseqs': data.get('cseqs', None),
                'img_path_dict': img_path_dict,
                'kc_vocab': kc_vocab
            }
            
            # RL 训练一步
            try:
                metrics = rl_trainer.train_step(batch_data, optimizer)
                epoch_rewards.append(metrics['reward'])
                epoch_losses.append(metrics['loss'])
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}: reward={metrics['reward']:.4f}, loss={metrics['loss']:.4f}")
            except Exception as e:
                print(f"  警告: 训练步骤失败: {e}")
                continue
        
        # 计算epoch平均指标
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        
        print(f"Epoch {epoch}/{args.num_epochs}: avg_reward={avg_reward:.4f}, avg_loss={avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_path = os.path.join(args.save_dir, f"cot_generator_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': cot_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': avg_reward,
                'loss': avg_loss
            }, save_path)
            print(f"  保存最佳模型: {save_path}")
    
    print(f"[RL] 训练完成！最佳奖励: {best_reward:.4f}")

if __name__ == "__main__":
    main()

