#!/usr/bin/env python3
"""
使用真实难度数据的难度排序正则化示例
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from crkt import CRKT

def example_with_real_difficulty():
    """使用真实难度数据的示例"""
    print("=== 使用真实难度数据的难度排序正则化示例 ===")
    
    # 配置参数 - 使用DBE_KT22数据集的难度信息
    config = {
        'num_q': 214,  # DBE_KT22数据集的题目数量
        'num_c': 100,
        'dim_qc': 128,
        'dim_difficulty': 64,
        'dim_knowledge': 128,
        'dropout': 0.1,
        'use_content_emb': False,  # 简化示例
        'use_analysis_emb': False,  # 简化示例
        'use_kc_emb': False,  # 简化示例
        'use_difficulty_ranking': True,  # 启用难度排序正则化
        'difficulty_ranking_weight': 0.1,  # 难度排序正则化权重 λ
        'difficulty_margin': 0.1,  # 难度排序的margin值
        'difficulty_csv_path': 'data/DBE_KT22/use_difficulty/gpt-4o_que_difficulty.csv',  # 真实难度数据
        'dataset_name': 'DBE_KT22'
    }
    
    print(f"配置: {config}")
    
    # 创建模型
    model = CRKT(config)
    print(f"模型创建成功: {type(model)}")
    
    # 测试前向传播
    batch_size, seq_len = 4, 20
    q = torch.randint(0, config['num_q'], (batch_size, seq_len))
    c = torch.randint(0, config['num_c'], (batch_size, seq_len, 3))
    r = torch.randint(0, 2, (batch_size, seq_len))
    q_shift = torch.randint(0, config['num_q'], (batch_size, seq_len))
    
    print(f"输入形状: q={q.shape}, c={c.shape}, r={r.shape}, q_shift={q_shift.shape}")
    
    # 前向传播
    with torch.no_grad():
        y, contrastive_loss, difficulty_ranking_loss = model.model(q, c, r, q_shift)
        print(f"输出形状: y={y.shape}")
        print(f"对比学习损失: {contrastive_loss:.6f}")
        print(f"难度排序正则化损失: {difficulty_ranking_loss:.6f}")
    
    # 测试训练步骤
    data = {
        'qseqs': q,
        'cseqs': c,
        'rseqs': r,
        'shft_qseqs': q_shift,
        'smasks': torch.ones(batch_size, seq_len, dtype=torch.bool),
        'shft_rseqs': torch.randint(0, 2, (batch_size, seq_len))
    }
    
    print("\n=== 测试训练步骤 ===")
    y, total_loss, main_loss, contrastive_loss, difficulty_ranking_loss = model.train_one_step(data)
    print(f"主损失: {main_loss:.6f}")
    print(f"对比学习损失: {contrastive_loss:.6f}")
    print(f"难度排序正则化损失: {difficulty_ranking_loss:.6f}")
    print(f"总损失: {total_loss:.6f}")
    
    # 分析难度分布
    print("\n=== 难度分布分析 ===")
    if model.model.difficulty_data is not None:
        difficulty_data = model.model.difficulty_data
        easy_count = (difficulty_data == 1).sum().item()
        medium_count = (difficulty_data == 2).sum().item()
        hard_count = (difficulty_data == 3).sum().item()
        
        print(f"Easy题目 (难度1): {easy_count}")
        print(f"Medium题目 (难度2): {medium_count}")
        print(f"Hard题目 (难度3): {hard_count}")
        
        # 显示一些具体的题目难度
        print("\n题目难度示例:")
        for i in range(min(10, config['num_q'])):
            difficulty = difficulty_data[i].item()
            difficulty_name = {1: 'Easy', 2: 'Medium', 3: 'Hard'}.get(difficulty, 'Unknown')
            print(f"  题目 {i}: {difficulty_name} (难度{difficulty})")
    
    print("\n=== 真实难度数据示例完成 ===")

if __name__ == "__main__":
    example_with_real_difficulty()
