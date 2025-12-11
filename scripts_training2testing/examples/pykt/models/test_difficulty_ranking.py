#!/usr/bin/env python3
"""
测试难度排序正则化功能的脚本
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from crkt import CRKT

def test_difficulty_ranking():
    """测试难度排序正则化功能"""
    print("=== 测试难度排序正则化功能 ===")
    
    # 配置参数
    config = {
        'num_q': 100,  # 减少题目数量以便测试
        'num_c': 50,
        'dim_qc': 64,
        'dim_difficulty': 32,
        'dim_knowledge': 64,
        'dropout': 0.1,
        'use_content_emb': False,  # 简化测试，不使用内容嵌入
        'use_analysis_emb': False,  # 简化测试，不使用解析嵌入
        'use_kc_emb': False,  # 简化测试，不使用KC嵌入
        'use_difficulty_ranking': True,  # 启用难度排序正则化
        'difficulty_ranking_weight': 0.1,  # 难度排序正则化权重 λ
        'difficulty_margin': 0.1,  # 难度排序的margin值
        'difficulty_csv_path': 'test_difficulty.csv',  # 测试用的难度数据文件
        'dataset_name': 'test'
    }
    
    print(f"配置: {config}")
    
    # 创建测试用的难度数据文件
    create_test_difficulty_csv('test_difficulty.csv', config['num_q'])
    
    # 创建模型
    model = CRKT(config)
    print(f"模型创建成功: {type(model)}")
    
    # 测试前向传播
    batch_size, seq_len = 2, 10
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
    
    # 测试梯度计算
    print("\n=== 测试梯度计算 ===")
    model.train()
    y, total_loss, main_loss, contrastive_loss, difficulty_ranking_loss = model.train_one_step(data)
    total_loss.backward()
    
    # 检查梯度
    print(f"主损失梯度范数: {main_loss.grad if main_loss.grad is not None else 'None'}")
    print(f"对比学习损失梯度范数: {contrastive_loss.grad if contrastive_loss.grad is not None else 'None'}")
    print(f"难度排序正则化损失梯度范数: {difficulty_ranking_loss.grad if difficulty_ranking_loss.grad is not None else 'None'}")
    
    # 清理测试文件
    if os.path.exists('test_difficulty.csv'):
        os.remove('test_difficulty.csv')
    
    print("\n=== 难度排序正则化功能测试完成 ===")

def create_test_difficulty_csv(filename, num_q):
    """创建测试用的难度数据文件"""
    import pandas as pd
    
    # 创建测试数据：前1/3为easy，中间1/3为medium，后1/3为hard
    data = []
    for i in range(num_q):
        if i < num_q // 3:
            difficulty = 1  # easy
        elif i < 2 * num_q // 3:
            difficulty = 2  # medium
        else:
            difficulty = 3  # hard
        data.append({'qid': i, 'difficulty': difficulty})
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"创建测试难度数据文件: {filename}")

if __name__ == "__main__":
    test_difficulty_ranking()
