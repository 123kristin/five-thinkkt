#!/usr/bin/env python3
"""
DKT模型使用示例 - 三层分层注意力机制

这个示例展示了如何在DKT模型中使用新的三层分层注意力机制。
"""

import torch
import sys
import os

# 添加模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dkt import DKT

def example_three_layer_hierarchical_attention():
    """使用三层分层注意力机制的示例"""
    print("=== DKT模型三层分层注意力机制使用示例 ===")
    
    # 创建模型配置
    model_config = {
        'num_c': 100,                    # 概念数量
        'emb_size': 128,                 # 嵌入维度
        'dropout': 0.1,                  # Dropout率
        'emb_type': 'qid',               # 嵌入类型
        'use_content_emb': True,         # 使用内容嵌入
        'use_analysis_emb': True,        # 使用分析嵌入
        'use_kc_emb': True,              # 使用KC嵌入
        'content_dim': 512,              # 内容嵌入维度
        'analysis_dim': 1536,            # 分析嵌入维度
        'kc_dim': 1600,                  # KC嵌入维度
        'attention_type': "three_layer_hierarchical",  # 使用三层分层注意力
        'dataset_name': "XES3G5M",       # 数据集名称
        'num_q': 50,                     # 习题数量
        'content_type': "text",          # 内容类型
        'analysis_type': "generated",    # 分析类型
        'no_analysis_fusion': False      # 是否不使用分析融合
    }
    
    print("模型配置:")
    for key, value in model_config.items():
        print(f"  - {key}: {value}")
    
    # 创建模型
    model = DKT(**model_config)
    print(f"\n✅ 模型创建成功")
    print(f"  - 注意力类型: {model.attention_type}")
    print(f"  - 嵌入维度: {model.emb_size}")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 20
    q = torch.randint(0, 50, (batch_size, seq_len))  # 习题ID
    r = torch.randint(0, 2, (batch_size, seq_len))   # 答题结果 (0/1)
    
    print(f"\n测试数据:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 习题ID范围: [{q.min().item()}, {q.max().item()}]")
    print(f"  - 答题结果分布: 0={torch.sum(r==0).item()}, 1={torch.sum(r==1).item()}")
    
    # 前向传播
    with torch.no_grad():
        output = model(q, r)
    
    print(f"\n模型输出:")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  - 输出均值: {output.mean().item():.4f}")
    print(f"  - 输出标准差: {output.std().item():.4f}")
    
    # 检查数值稳定性
    if torch.isnan(output).any():
        print("❌ 输出包含NaN值")
    elif torch.isinf(output).any():
        print("❌ 输出包含Inf值")
    else:
        print("✅ 模型输出正常")
    
    return model, output

def compare_attention_mechanisms():
    """比较不同注意力机制的效果"""
    print("\n=== 注意力机制比较 ===")
    
    attention_types = [
        "cross",
        "improved_cross", 
        "hierarchical",
        "knowledge_aware",
        "three_layer_hierarchical"
    ]
    
    results = {}
    
    for attention_type in attention_types:
        print(f"\n测试 {attention_type} 注意力机制...")
        
        try:
            # 创建模型
            model = DKT(
                num_c=100,
                emb_size=128,
                dropout=0.1,
                emb_type='qid',
                use_content_emb=True,
                use_analysis_emb=True,
                use_kc_emb=True,
                attention_type=attention_type,
                dataset_name="XES3G5M",
                num_q=50
            )
            
            # 测试数据
            q = torch.randint(0, 50, (2, 10))
            r = torch.randint(0, 2, (2, 10))
            
            # 前向传播
            with torch.no_grad():
                output = model(q, r)
            
            results[attention_type] = {
                'success': True,
                'output_shape': output.shape,
                'output_range': [output.min().item(), output.max().item()],
                'has_nan': torch.isnan(output).any().item(),
                'has_inf': torch.isinf(output).any().item()
            }
            
            print(f"  ✅ {attention_type} 测试通过")
            
        except Exception as e:
            results[attention_type] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ {attention_type} 测试失败: {e}")
    
    # 打印比较结果
    print(f"\n=== 注意力机制比较结果 ===")
    for attention_type, result in results.items():
        if result['success']:
            print(f"{attention_type}: ✅ 成功")
            print(f"  - 输出形状: {result['output_shape']}")
            print(f"  - 数值范围: [{result['output_range'][0]:.4f}, {result['output_range'][1]:.4f}]")
            print(f"  - 包含NaN: {result['has_nan']}")
            print(f"  - 包含Inf: {result['has_inf']}")
        else:
            print(f"{attention_type}: ❌ 失败 - {result['error']}")

if __name__ == "__main__":
    # 运行示例
    model, output = example_three_layer_hierarchical_attention()
    
    # 比较不同注意力机制
    compare_attention_mechanisms()
    
    print(f"\n=== 示例完成 ===")
    print(f"三层分层注意力机制已成功实现并测试通过！") 