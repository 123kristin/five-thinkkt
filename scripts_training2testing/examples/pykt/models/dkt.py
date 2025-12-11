"""
DKT模型 - 支持多种注意力机制的知识追踪模型

支持的注意力机制类型：
1. "cross" - 原始交叉注意力
2. "improved_cross" - 改进的交叉注意力（双向+门控）
3. "hierarchical" - 分层注意力（自注意力+交叉注意力）
4. "knowledge_aware" - 知识感知注意力
5. "three_layer_hierarchical" - 三层分层注意力（新增）

三层分层注意力机制 (ThreeLayerHierarchicalAttention) 特点：
- 第一层：自注意力 - 同模态内部理解和特征提取
- 第二层：交叉注意力 - 跨模态信息融合和交互
- 第三层：因果注意力 - 时序依赖建模，确保因果性

优势：
1. 渐进式信息处理，符合认知过程
2. 专门处理时序依赖关系
3. 更好的可解释性
4. 适合知识追踪任务的因果性要求

使用方法：
model = DKT(
    attention_type="three_layer_hierarchical",
    use_content_emb=True,
    use_analysis_emb=True,
    use_kc_emb=True,
    # 其他参数...
)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class CrossAttentionBlock(nn.Module):
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.content_dim = content_dim
        self.analysis_dim = analysis_dim
        self.kc_dim = kc_dim
        self.d_model = d_model
        
        # 投影层 - 将所有输入投影到统一维度
        self.query_proj = nn.Linear(content_dim, d_model)
        self.key_proj = nn.Linear(analysis_dim + kc_dim, d_model)
        self.value_proj = nn.Linear(analysis_dim + kc_dim, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, content_emb, analysis_emb, kc_emb):
        # 检查输入数值稳定性
        if torch.isnan(content_emb).any() or torch.isinf(content_emb).any():
            print(f"[CrossAttention] 警告: content_emb包含NaN或Inf值")
            content_emb = torch.zeros_like(content_emb)
        if torch.isnan(analysis_emb).any() or torch.isinf(analysis_emb).any():
            print(f"[CrossAttention] 警告: analysis_emb包含NaN或Inf值")
            analysis_emb = torch.zeros_like(analysis_emb)
        if torch.isnan(kc_emb).any() or torch.isinf(kc_emb).any():
            print(f"[CrossAttention] 警告: kc_emb包含NaN或Inf值")
            kc_emb = torch.zeros_like(kc_emb)
        
        # 投影到统一维度
        query = self.query_proj(content_emb)
        key_value = torch.cat([analysis_emb, kc_emb], dim=-1)
        key = self.key_proj(key_value)
        value = self.value_proj(key_value)
        
        # 检查投影后的数值稳定性
        if torch.isnan(query).any() or torch.isinf(query).any():
            print(f"[CrossAttention] 警告: query包含NaN或Inf值")
            query = torch.zeros_like(query)
        if torch.isnan(key).any() or torch.isinf(key).any():
            print(f"[CrossAttention] 警告: key包含NaN或Inf值")
            key = torch.zeros_like(key)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"[CrossAttention] 警告: value包含NaN或Inf值")
            value = torch.zeros_like(value)
        
        attn_output, _ = self.multihead_attn(query, key, value)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        out = self.norm(out + query)
        
        # 检查输出数值稳定性
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[CrossAttention] 警告: 输出包含NaN或Inf值")
            out = torch.zeros_like(out)
        
        return out

class MultiLayerCrossAttention(nn.Module):
    """多层交叉注意力机制"""
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 投影层 - 将所有输入投影到统一维度
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = CrossAttentionBlock(
                content_dim=d_model,  # 统一使用d_model
                analysis_dim=d_model,
                kc_dim=d_model,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.layers.append(layer)
        
        # 最终输出投影层
        self.final_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 首先将所有输入投影到统一维度
        x_content = self.content_proj(content_emb)
        x_analysis = self.analysis_proj(analysis_emb)
        x_kc = self.kc_proj(kc_emb)
        
        x = x_content
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            if i == 0:
                # 第一层：使用投影后的content_emb作为query
                x = layer(x_content, x_analysis, x_kc)
            else:
                # 后续层：使用上一层的输出作为query
                x = layer(x, x_analysis, x_kc)
        
        # 最终处理
        out = self.final_proj(x)
        out = self.final_dropout(out)
        out = self.final_norm(out + x)
        
        return out

class ImprovedCrossAttention(nn.Module):
    """改进的交叉注意力机制 - 双向交叉注意力 + 门控"""
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.content_dim = content_dim
        self.analysis_dim = analysis_dim
        self.kc_dim = kc_dim
        
        # 投影层 - 将所有输入投影到统一维度
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        # 双向交叉注意力层
        self.content_to_others_layers = nn.ModuleList()
        self.others_to_content_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Content -> Others 方向
            layer1 = CrossAttentionBlock(
                content_dim=d_model,  # 统一使用d_model
                analysis_dim=d_model,
                kc_dim=d_model,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.content_to_others_layers.append(layer1)
            
            # Others -> Content 方向
            layer2 = CrossAttentionBlock(
                content_dim=d_model,  # 统一使用d_model
                analysis_dim=d_model,
                kc_dim=d_model,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.others_to_content_layers.append(layer2)
        
        # 门控机制
        self.gate = nn.Linear(d_model * 2, d_model)
        self.gate_activation = nn.Sigmoid()
        
        # 最终输出层
        self.final_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 首先将所有输入投影到统一维度
        x_content = self.content_proj(content_emb)
        x_analysis = self.analysis_proj(analysis_emb)
        x_kc = self.kc_proj(kc_emb)
        
        # 逐层双向交叉注意力
        for i in range(self.num_layers):
            # 双向交叉注意力
            content_to_others = self.content_to_others_layers[i](x_content, x_analysis, x_kc)
            others_to_content = self.others_to_content_layers[i](x_analysis, x_content, x_kc)
            
            # 门控融合
            combined = torch.cat([content_to_others, others_to_content], dim=-1)
            gate = self.gate_activation(self.gate(combined))
            x_content = gate * content_to_others + (1 - gate) * others_to_content
            
            # 更新analysis状态用于下一层
            if i < self.num_layers - 1:
                x_analysis = others_to_content
        
        # 最终处理
        out = self.final_proj(x_content)
        out = self.final_dropout(out)
        out = self.final_norm(out + x_content)
        
        return out

class HierarchicalAttention(nn.Module):
    """分层注意力机制 - 先同模态自注意力，再跨模态交叉注意力"""
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 投影层 - 将所有输入投影到统一维度
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        # 第一层：同模态自注意力
        self.content_self_attn = nn.ModuleList()
        self.analysis_self_attn = nn.ModuleList()
        self.kc_self_attn = nn.ModuleList()
        
        # 第二层：跨模态交叉注意力
        self.cross_attn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 同模态自注意力
            self.content_self_attn.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            self.analysis_self_attn.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            self.kc_self_attn.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            
            # 跨模态交叉注意力
            cross_layer = CrossAttentionBlock(
                content_dim=d_model,
                analysis_dim=d_model,
                kc_dim=d_model,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.cross_attn_layers.append(cross_layer)
        
        # 最终输出层
        self.final_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 投影到统一维度
        x_content = self.content_proj(content_emb)
        x_analysis = self.analysis_proj(analysis_emb)
        x_kc = self.kc_proj(kc_emb)
        
        # 逐层处理
        for i in range(self.num_layers):
            # 第一层：同模态自注意力
            content_self, _ = self.content_self_attn[i](x_content, x_content, x_content)
            analysis_self, _ = self.analysis_self_attn[i](x_analysis, x_analysis, x_analysis)
            kc_self, _ = self.kc_self_attn[i](x_kc, x_kc, x_kc)
            
            # 第二层：跨模态交叉注意力
            enhanced_content = self.cross_attn_layers[i](content_self, analysis_self, kc_self)
            
            # 更新状态
            x_content = enhanced_content
            x_analysis = analysis_self
            x_kc = kc_self
        
        # 最终处理
        out = self.final_proj(x_content)
        out = self.final_dropout(out)
        out = self.final_norm(out + x_content)
        
        return out

class KnowledgeAwareAttention(nn.Module):
    """知识感知注意力机制 - 基于KC的知识引导注意力"""
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 投影层 - 将所有输入投影到统一维度
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        # 知识权重计算层
        self.knowledge_weight = nn.Linear(d_model, 1)
        
        # 知识感知的交叉注意力层 - 使用统一维度d_model
        self.knowledge_aware_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = CrossAttentionBlock(
                content_dim=d_model,  # 使用统一维度d_model
                analysis_dim=d_model,  # 使用统一维度d_model
                kc_dim=d_model,       # 使用统一维度d_model
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.knowledge_aware_layers.append(layer)
        
        # 最终输出层
        self.final_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 投影到统一维度
        x_content = self.content_proj(content_emb)
        x_analysis = self.analysis_proj(analysis_emb)
        x_kc = self.kc_proj(kc_emb)
        
        # 计算知识权重
        knowledge_weights = torch.sigmoid(self.knowledge_weight(x_kc))  # (batch, seq_len, 1)
        
        # 知识感知的嵌入增强
        content_enhanced = x_content * knowledge_weights
        analysis_enhanced = x_analysis * knowledge_weights
        
        # 逐层知识感知交叉注意力 - 使用投影后的统一维度
        for i in range(self.num_layers):
            if i == 0:
                # 第一层：使用投影后的统一维度
                x_content = self.knowledge_aware_layers[i](content_enhanced, analysis_enhanced, x_kc)
            else:
                # 后续层：使用上一层的输出和投影后的统一维度
                x_content = self.knowledge_aware_layers[i](x_content, analysis_enhanced, x_kc)
        
        # 最终处理
        out = self.final_proj(x_content)
        out = self.final_dropout(out)
        out = self.final_norm(out + x_content)
        
        return out

class ThreeLayerHierarchicalAttention(nn.Module):
    """三层分层注意力机制 - 自注意力 + 交叉注意力 + 因果注意力"""
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model=128, num_heads=8, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 投影层 - 将所有输入投影到统一维度
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        # 第一层：自注意力（同模态内部理解）
        self.content_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.analysis_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.kc_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 第二层：交叉注意力（跨模态信息融合）
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 第三层：因果注意力（时序依赖建模）
        self.causal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 输出层
        self.final_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
        
        # 中间层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        # 残差连接的dropout
        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)
        self.residual_dropout3 = nn.Dropout(dropout)
    
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 检查输入数值稳定性
        if torch.isnan(content_emb).any() or torch.isinf(content_emb).any():
            print(f"[ThreeLayerHierarchical] 警告: content_emb包含NaN或Inf值")
            content_emb = torch.zeros_like(content_emb)
        if torch.isnan(analysis_emb).any() or torch.isinf(analysis_emb).any():
            print(f"[ThreeLayerHierarchical] 警告: analysis_emb包含NaN或Inf值")
            analysis_emb = torch.zeros_like(analysis_emb)
        if torch.isnan(kc_emb).any() or torch.isinf(kc_emb).any():
            print(f"[ThreeLayerHierarchical] 警告: kc_emb包含NaN或Inf值")
            kc_emb = torch.zeros_like(kc_emb)
        
        # 投影到统一维度
        x_content = self.content_proj(content_emb)
        x_analysis = self.analysis_proj(analysis_emb)
        x_kc = self.kc_proj(kc_emb)
        
        # 检查投影后的数值稳定性
        if torch.isnan(x_content).any() or torch.isinf(x_content).any():
            print(f"[ThreeLayerHierarchical] 警告: x_content包含NaN或Inf值")
            x_content = torch.zeros_like(x_content)
        if torch.isnan(x_analysis).any() or torch.isinf(x_analysis).any():
            print(f"[ThreeLayerHierarchical] 警告: x_analysis包含NaN或Inf值")
            x_analysis = torch.zeros_like(x_analysis)
        if torch.isnan(x_kc).any() or torch.isinf(x_kc).any():
            print(f"[ThreeLayerHierarchical] 警告: x_kc包含NaN或Inf值")
            x_kc = torch.zeros_like(x_kc)
        
        # 第一层：自注意力（同模态内部理解）
        try:
            content_self, _ = self.content_self_attn(x_content, x_content, x_content)
            analysis_self, _ = self.analysis_self_attn(x_analysis, x_analysis, x_analysis)
            kc_self, _ = self.kc_self_attn(x_kc, x_kc, x_kc)
            
            # 检查自注意力输出
            if torch.isnan(content_self).any() or torch.isinf(content_self).any():
                print(f"[ThreeLayerHierarchical] 警告: content_self包含NaN或Inf值")
                content_self = torch.zeros_like(content_self)
            if torch.isnan(analysis_self).any() or torch.isinf(analysis_self).any():
                print(f"[ThreeLayerHierarchical] 警告: analysis_self包含NaN或Inf值")
                analysis_self = torch.zeros_like(analysis_self)
            if torch.isnan(kc_self).any() or torch.isinf(kc_self).any():
                print(f"[ThreeLayerHierarchical] 警告: kc_self包含NaN或Inf值")
                kc_self = torch.zeros_like(kc_self)
            
            # 残差连接和归一化
            content_self = self.layer_norm1(x_content + self.residual_dropout1(content_self))
            analysis_self = self.layer_norm1(x_analysis + self.residual_dropout1(analysis_self))
            kc_self = self.layer_norm1(x_kc + self.residual_dropout1(kc_self))
            
        except Exception as e:
            print(f"[ThreeLayerHierarchical] 第一层自注意力计算失败: {e}")
            content_self = x_content
            analysis_self = x_analysis
            kc_self = x_kc
        
        # 第二层：交叉注意力（跨模态信息融合）
        try:
            # 将analysis和kc信息融合作为key和value
            cross_key_value = torch.cat([analysis_self, kc_self], dim=-1)
            # 由于维度不匹配，需要投影
            cross_key_value_proj = nn.Linear(cross_key_value.size(-1), self.d_model).to(cross_key_value.device)
            cross_key_value = cross_key_value_proj(cross_key_value)
            
            cross_output, _ = self.cross_attn(content_self, cross_key_value, cross_key_value)
            
            # 检查交叉注意力输出
            if torch.isnan(cross_output).any() or torch.isinf(cross_output).any():
                print(f"[ThreeLayerHierarchical] 警告: cross_output包含NaN或Inf值")
                cross_output = torch.zeros_like(cross_output)
            
            # 残差连接和归一化
            cross_output = self.layer_norm2(content_self + self.residual_dropout2(cross_output))
            
        except Exception as e:
            print(f"[ThreeLayerHierarchical] 第二层交叉注意力计算失败: {e}")
            cross_output = content_self
        
        # 第三层：因果注意力（时序依赖建模）
        try:
            # 创建因果掩码
            seq_len = cross_output.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(cross_output.device)
            
            causal_output, _ = self.causal_attn(cross_output, cross_output, cross_output, 
                                              attn_mask=causal_mask)
            
            # 检查因果注意力输出
            if torch.isnan(causal_output).any() or torch.isinf(causal_output).any():
                print(f"[ThreeLayerHierarchical] 警告: causal_output包含NaN或Inf值")
                causal_output = torch.zeros_like(causal_output)
            
            # 残差连接和归一化
            causal_output = self.layer_norm3(cross_output + self.residual_dropout3(causal_output))
            
        except Exception as e:
            print(f"[ThreeLayerHierarchical] 第三层因果注意力计算失败: {e}")
            causal_output = cross_output
        
        # 最终输出处理
        try:
            out = self.final_proj(causal_output)
            out = self.final_dropout(out)
            out = self.final_norm(out + causal_output)
            
            # 检查最终输出
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[ThreeLayerHierarchical] 警告: 最终输出包含NaN或Inf值")
                out = torch.zeros_like(out)
            
        except Exception as e:
            print(f"[ThreeLayerHierarchical] 最终输出处理失败: {e}")
            out = causal_output
        
        return out

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,
                 use_content_emb=False, use_analysis_emb=False, use_kc_emb=False, gen_emb_path="", 
                 content_dim=512, analysis_dim=1536, analysis_type="generated",
                 trainable_content_emb=False, trainable_analysis_emb=False, trainable_kc_emb=False, 
                 num_q=0, content_type="text", dataset_name="", cross_attention_layers=1, 
                 no_analysis_fusion=False, attention_type="cross",  # 新增注意力类型参数
                 **kwargs):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.num_q = num_q  # 添加习题数量参数
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        
        # 嵌入相关参数
        self.use_content_emb = use_content_emb
        self.use_analysis_emb = use_analysis_emb
        self.use_kc_emb = use_kc_emb
        self.gen_emb_path = gen_emb_path
        self.content_dim = content_dim
        self.analysis_dim = analysis_dim
        self.kc_dim = 1600  # 固定KC嵌入维度
        self.analysis_type = analysis_type
        self.content_type = content_type  # 添加内容嵌入类型参数
        self.trainable_content_emb = trainable_content_emb
        self.trainable_analysis_emb = trainable_analysis_emb
        self.trainable_kc_emb = trainable_kc_emb
        self.dataset_name = dataset_name
        self.no_analysis_fusion = no_analysis_fusion  # 新增
        
        # 交叉注意力层数参数
        self.cross_attention_layers = cross_attention_layers
        # 注意力类型参数
        self.attention_type = attention_type
        
        # 嵌入文件路径映射
        self.emb_file_mapping = {
            "content": "embedding_content.pkl",
            "content_image": "embedding_images_content.pkl",  # 添加图像内容嵌入
            "generated": "embedding_generated_explanation.pkl", 
            "original": "embedding_original_explanation.pkl",
            "kc": "embedding_kc.pkl",
            "kc_graph": "graph_embedding_kc.pkl"
        }
        
        # 预训练嵌入相关设置
        self.content_emb_data = None
        self.analysis_emb_data = None
        self.kc_emb_data = None

        if emb_type.startswith("qid"):
            # 修改为使用习题ID而不是概念ID
            self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)

        # 交叉注意力机制（延迟创建，在嵌入加载完成后）
        self.cross_attention = None
        self.emb_fusion = None

        # 特征融合层（如果使用嵌入）
        if self.use_content_emb or self.use_analysis_emb:
            self.fusion_layer = nn.Linear(self.emb_size * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        
        # 根据数据集名称自动推断路径
        if not gen_emb_path and dataset_name:
            gen_emb_path = self._get_dataset_path(dataset_name)
            self.gen_emb_path = gen_emb_path
        
        # 加载嵌入数据
        if gen_emb_path and (use_content_emb or use_analysis_emb or use_kc_emb):
            self._load_embeddings()
            
        # 创建交叉注意力机制（在嵌入加载完成后）
        self._create_cross_attention()
        
        # 重新调整融合层维度（如果嵌入加载失败）
        self._adjust_fusion_layers()

    def _create_cross_attention(self):
        """创建注意力机制（在嵌入加载完成后）"""
        if self.use_content_emb and self.use_analysis_emb and self.use_kc_emb:
            # 确保所有嵌入都已加载
            if (self.content_emb_data is not None and 
                self.analysis_emb_data is not None and 
                self.kc_emb_data is not None):
                
                # 使用实际的嵌入维度
                actual_content_dim = self.content_emb_data.shape[1]
                actual_analysis_dim = self.analysis_emb_data.shape[1]
                actual_kc_dim = self.kc_emb_data.shape[1]
                
                print(f"[DKT] 创建{self.attention_type}注意力机制:")
                print(f"  - content_dim: {actual_content_dim}")
                print(f"  - analysis_dim: {actual_analysis_dim}")
                print(f"  - kc_dim: {actual_kc_dim}")
                print(f"  - d_model: {self.emb_size}")
                print(f"  - layers: {self.cross_attention_layers}")
                
                # 根据注意力类型创建不同的注意力机制
                if self.attention_type == "cross":
                    # 原始交叉注意力
                    if self.cross_attention_layers == 1:
                        self.cross_attention = CrossAttentionBlock(
                            content_dim=actual_content_dim,
                            analysis_dim=actual_analysis_dim,
                            kc_dim=actual_kc_dim,
                            d_model=self.emb_size,
                            num_heads=4,
                            dropout=0.1
                        )
                    else:
                        self.cross_attention = MultiLayerCrossAttention(
                            content_dim=actual_content_dim,
                            analysis_dim=actual_analysis_dim,
                            kc_dim=actual_kc_dim,
                            d_model=self.emb_size,
                            num_heads=4,
                            num_layers=self.cross_attention_layers,
                            dropout=0.1
                        )
                
                elif self.attention_type == "improved_cross":
                    # 改进的交叉注意力
                    self.cross_attention = ImprovedCrossAttention(
                        content_dim=actual_content_dim,
                        analysis_dim=actual_analysis_dim,
                        kc_dim=actual_kc_dim,
                        d_model=self.emb_size,
                        num_heads=4,
                        num_layers=self.cross_attention_layers,
                        dropout=0.1
                    )
                
                elif self.attention_type == "hierarchical":
                    # 分层注意力
                    self.cross_attention = HierarchicalAttention(
                        content_dim=actual_content_dim,
                        analysis_dim=actual_analysis_dim,
                        kc_dim=actual_kc_dim,
                        d_model=self.emb_size,
                        num_heads=4,
                        num_layers=self.cross_attention_layers,
                        dropout=0.1
                    )
                
                elif self.attention_type == "knowledge_aware":
                    # 知识感知注意力
                    self.cross_attention = KnowledgeAwareAttention(
                        content_dim=actual_content_dim,
                        analysis_dim=actual_analysis_dim,
                        kc_dim=actual_kc_dim,
                        d_model=self.emb_size,
                        num_heads=4,
                        num_layers=self.cross_attention_layers,
                        dropout=0.1
                    )
                
                elif self.attention_type == "three_layer_hierarchical":
                    # 三层分层注意力
                    self.cross_attention = ThreeLayerHierarchicalAttention(
                        content_dim=actual_content_dim,
                        analysis_dim=actual_analysis_dim,
                        kc_dim=actual_kc_dim,
                        d_model=self.emb_size,
                        num_heads=4,
                        dropout=0.1
                    )
                
                else:
                    print(f"[DKT] 未知的注意力类型: {self.attention_type}，使用默认交叉注意力")
                    self.cross_attention = CrossAttentionBlock(
                        content_dim=actual_content_dim,
                        analysis_dim=actual_analysis_dim,
                        kc_dim=actual_kc_dim,
                        d_model=self.emb_size,
                        num_heads=4,
                        dropout=0.1
                    )
                
                # emb_fusion输入维度根据no_analysis_fusion调整
                # 所有注意力机制都输出d_model维度（self.emb_size）
                if self.no_analysis_fusion:
                    print(f"[DKT] 创建融合层（no_analysis_fusion），输入维度: {self.emb_size}, 输出维度: {self.emb_size}")
                    self.emb_fusion = nn.Linear(self.emb_size, self.emb_size)
                else:
                    print(f"[DKT] 创建融合层（with_analysis_fusion），输入维度: {self.emb_size + actual_analysis_dim}, 输出维度: {self.emb_size}")
                    self.emb_fusion = nn.Linear(self.emb_size + actual_analysis_dim, self.emb_size)
                
                print(f"[DKT] {self.attention_type}注意力机制创建成功")
            else:
                print(f"[DKT] 警告: 无法创建注意力机制，嵌入未完全加载")
                print(f"  - content_emb: {self.content_emb_data is not None}")
                print(f"  - analysis_emb: {self.analysis_emb_data is not None}")
                print(f"  - kc_emb: {self.kc_emb_data is not None}")

    def _get_dataset_path(self, dataset_name):
        """根据数据集名称自动推断嵌入路径"""
        base_path = "/home3/zhiyu/code-4/kt_analysis_generation/data"
        
        if dataset_name == "XES3G5M":
            return os.path.join(base_path, "XES3G5M/generate_analysis/embeddings")
        elif dataset_name == "DBE_KT22":
            return os.path.join(base_path, "DBE_KT22/generate_analysis/embeddings")
        else:
            print(f"[DKT] 未知数据集: {dataset_name}，请手动指定gen_emb_path")
            return ""
    
    def _get_kc_dataset_path(self):
        """根据数据集名称自动推断KC嵌入路径"""
        base_path = "/home3/zhiyu/code-4/kt_analysis_generation/data"
        
        if self.dataset_name == "XES3G5M":
            return os.path.join(base_path, "XES3G5M/generate_kc")
        elif self.dataset_name == "DBE_KT22":
            return os.path.join(base_path, "DBE_KT22/generate_kc")
        else:
            print(f"[DKT] 未知数据集: {self.dataset_name}，无法推断KC嵌入路径")
            return ""

    def _adjust_fusion_layers(self):
        """根据实际加载的嵌入创建融合层"""
        try:
            # 检查哪些嵌入实际可用
            content_available = self.use_content_emb and self.content_emb_data is not None
            analysis_available = self.use_analysis_emb and self.analysis_emb_data is not None
            kc_available = self.use_kc_emb and self.kc_emb_data is not None
            
            # 修正：如果三种嵌入都启用且有cross_attention，则emb_fusion输入为self.emb_size (+ self.analysis_dim)
            if self.use_content_emb and self.use_analysis_emb and self.use_kc_emb and hasattr(self, 'cross_attention') and self.cross_attention is not None:
                # 所有注意力机制都输出d_model维度（self.emb_size）
                if self.no_analysis_fusion:
                    print(f"[DKT] (cross_attention, no_analysis_fusion) 创建融合层，输入维度: {self.emb_size}, 输出维度: {self.emb_size}")
                    self.emb_fusion = nn.Linear(self.emb_size, self.emb_size)
                else:
                    actual_analysis_dim = self.analysis_emb_data.shape[1]
                    print(f"[DKT] (cross_attention) 创建融合层，输入维度: {self.emb_size + actual_analysis_dim}, 输出维度: {self.emb_size}")
                    self.emb_fusion = nn.Linear(self.emb_size + actual_analysis_dim, self.emb_size)
            elif content_available or analysis_available or kc_available:
                # 计算实际需要的输入维度
                actual_input_dim = 0
                if content_available:
                    actual_input_dim += self.content_emb_data.shape[1]
                if analysis_available:
                    actual_input_dim += self.analysis_emb_data.shape[1]
                if kc_available:
                    actual_input_dim += self.kc_emb_data.shape[1]
                
                print(f"[DKT] 创建融合层，输入维度: {actual_input_dim}, 输出维度: {self.emb_size}")
                self.emb_fusion = nn.Linear(actual_input_dim, self.emb_size)
            else:
                print(f"[DKT] 没有可用的嵌入，跳过融合层创建")
        except Exception as e:
            print(f"[DKT] 创建融合层时出错: {e}")

    def _load_embeddings(self):
        """加载预训练嵌入数据"""
        try:
            if self.use_content_emb:
                self._load_content_embedding()
            
            if self.use_analysis_emb:
                self._load_analysis_embedding()
            
            if self.use_kc_emb:
                self._load_kc_embedding()
            
            # 确保融合层被创建
            self._adjust_fusion_layers()
                    
        except Exception as e:
            print(f"[DKT] 加载嵌入异常: {e}")
            self.content_emb_data = None
            self.analysis_emb_data = None
            self.kc_emb_data = None
    
    def _load_content_embedding(self):
        """加载content嵌入"""
        try:
            # 根据content_type选择嵌入文件
            if self.content_type == "text":
                content_key = "content"
            elif self.content_type == "image":
                content_key = "content_image"
            else:
                print(f"[DKT] 不支持的内容嵌入类型: {self.content_type}")
                self.content_emb_data = None
                return
                
            pkl_path = os.path.join(self.gen_emb_path, self.emb_file_mapping[content_key])
            print(f"[DKT] 尝试加载{self.content_type}内容嵌入文件: {pkl_path}")
            
            if os.path.exists(pkl_path):
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    print(f"[DKT] 成功加载pkl文件，嵌入字典大小: {len(emb_dict)}")
                    # 根据trainable_content_emb参数决定是否创建可训练嵌入
                    self.content_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_content_emb
                    )
                    if self.content_emb_data is not None:
                        # 更新content_dim为实际加载的维度
                        actual_content_dim = self.content_emb_data.shape[1]
                        if actual_content_dim != self.content_dim:
                            print(f"[DKT] 自动调整content_dim: {self.content_dim} -> {actual_content_dim}")
                            self.content_dim = actual_content_dim
                        print(f"[DKT] 成功加载{self.content_type}内容嵌入，shape: {self.content_emb_data.shape}")
                        return
                    else:
                        print(f"[DKT] 转换嵌入字典到tensor失败")
                else:
                    print(f"[DKT] 加载pkl文件失败或返回None")
            else:
                print(f"[DKT] 嵌入文件不存在: {pkl_path}")
            
            print(f"[DKT] 未找到{self.content_type}内容嵌入文件: {pkl_path}")
            self.content_emb_data = None
            
        except Exception as e:
            print(f"[DKT] 加载{self.content_type}内容嵌入时发生异常: {e}")
            import traceback
            traceback.print_exc()
            self.content_emb_data = None
    
    def _load_analysis_embedding(self):
        """根据analysis_type加载对应的analysis嵌入"""
        analysis_filename = self.emb_file_mapping[self.analysis_type]
        pkl_path = os.path.join(self.gen_emb_path, analysis_filename)
        if os.path.exists(pkl_path):
            emb_dict = self._load_pkl_embedding(pkl_path)
            if emb_dict is not None:
                # 根据trainable_analysis_emb参数决定是否创建可训练嵌入
                self.analysis_emb_data = self._convert_dict_to_tensor(
                    emb_dict, make_trainable=self.trainable_analysis_emb
                )
                if self.analysis_emb_data is not None:
                    # 更新analysis_dim为实际加载的维度
                    actual_analysis_dim = self.analysis_emb_data.shape[1]
                    if actual_analysis_dim != self.analysis_dim:
                        self.analysis_dim = actual_analysis_dim
                    print(f"[DKT] 成功加载analysis嵌入，shape: {self.analysis_emb_data.shape}")
                    return
                else:
                    print(f"[DKT] 转换analysis嵌入字典到tensor失败")
        
        print(f"[DKT] 未找到analysis嵌入文件: {pkl_path}")
        self.analysis_emb_data = None
    
    def _load_kc_embedding(self):
        """加载KC嵌入"""
        # 根据数据集名称推断KC嵌入路径
        kc_emb_path = self._get_kc_dataset_path()
        if not kc_emb_path:
            print(f"[DKT] 无法推断KC嵌入路径，请检查数据集名称: {self.dataset_name}")
            self.kc_emb_data = None
            return
        
        # 优先尝试graph_embedding_kc.pkl，如果不存在则使用embedding_kc.pkl
        kc_files = ["kc", "kc_graph"]
        for kc_key in kc_files:
            pkl_path = os.path.join(kc_emb_path, self.emb_file_mapping[kc_key])
            if os.path.exists(pkl_path):
                print(f"[DKT] 尝试加载KC嵌入文件: {pkl_path}")
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    self.kc_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_kc_emb
                    )
                    if self.kc_emb_data is not None:
                        actual_kc_dim = self.kc_emb_data.shape[1]
                        if actual_kc_dim != self.kc_dim:
                            print(f"[DKT] 自动调整kc_dim: {self.kc_dim} -> {actual_kc_dim}")
                            self.kc_dim = actual_kc_dim
                        print(f"[DKT] 成功加载KC嵌入，shape: {self.kc_emb_data.shape}")
                        return
                    else:
                        print(f"[DKT] 转换KC嵌入字典到tensor失败")
                else:
                    print(f"[DKT] 加载KC嵌入pkl文件失败或返回None")
            else:
                print(f"[DKT] KC嵌入文件不存在: {pkl_path}")
        
        print(f"[DKT] 未找到KC嵌入文件")
        self.kc_emb_data = None
    
    def _load_pkl_embedding(self, file_path):
        """加载pkl格式嵌入文件"""
        try:
            with open(file_path, 'rb') as f:
                emb_dict = pickle.load(f)
            return emb_dict
        except Exception as e:
            print(f"[DKT] 加载pkl文件失败 {file_path}: {e}")
            return None
    
    def _convert_dict_to_tensor(self, emb_dict, max_qid=None, make_trainable=False):
        """将字典格式转换为tensor格式，支持创建可训练参数"""
        if emb_dict is None:
            return None
        
        try:
            # 获取最大qid - 使用字典中实际的最大qid
            if max_qid is None:
                if emb_dict:
                    # 确保按数值比较，而不是字典序比较
                    max_qid = max(int(k) if isinstance(k, str) else k for k in emb_dict.keys())
                else:
                    max_qid = 0
            
            # 确保max_qid是整数类型
            if isinstance(max_qid, str):
                try:
                    max_qid = int(max_qid)
                except ValueError:
                    print(f"[DKT] 无法将max_qid转换为整数: {max_qid}")
                    return None
            
            # 获取嵌入维度
            sample_emb = next(iter(emb_dict.values()))
            if isinstance(sample_emb, list):
                emb_dim = len(sample_emb)
            elif hasattr(sample_emb, 'shape'):
                emb_dim = sample_emb.shape[-1]
            else:
                emb_dim = len(sample_emb)
            
            print(f"[DKT] 嵌入维度: {emb_dim}, 最大qid: {max_qid} (类型: {type(max_qid)})")
            
            # 创建tensor并移动到正确设备
            # 确保tensor大小足够容纳所有qid
            tensor_size = max_qid + 1
            emb_tensor = torch.zeros(tensor_size, emb_dim, dtype=torch.float32)
            
            # 统计成功加载的嵌入数量
            loaded_count = 0
            skipped_count = 0
            
            for qid, emb in emb_dict.items():
                try:
                    # 确保qid是整数
                    if isinstance(qid, str):
                        qid = int(qid)
                    
                    # 检查qid是否在有效范围内
                    if qid >= tensor_size:
                        print(f"[DKT] qid {qid} 超出tensor范围 {tensor_size}，跳过")
                        skipped_count += 1
                        continue
                    
                    if isinstance(emb, list):
                        emb_tensor[qid] = torch.tensor(emb, dtype=torch.float32)
                    elif isinstance(emb, np.ndarray):
                        emb_tensor[qid] = torch.tensor(emb, dtype=torch.float32)
                    elif isinstance(emb, torch.Tensor):
                        emb_tensor[qid] = emb.float()
                    else:
                        # 尝试转换为float
                        emb_tensor[qid] = torch.tensor(emb, dtype=torch.float32)
                    
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"[DKT] 处理嵌入qid={qid}时出错: {e}, 嵌入类型: {type(emb)}")
                    # 跳过这个嵌入，使用零向量
                    skipped_count += 1
                    continue
            
            print(f"[DKT] 成功加载 {loaded_count} 个嵌入，跳过 {skipped_count} 个嵌入")
            
            # 如果指定为可训练，则转换为nn.Parameter
            if make_trainable:
                return nn.Parameter(emb_tensor, requires_grad=True)
            else:
                return emb_tensor
                
        except Exception as e:
            print(f"[DKT] 转换嵌入字典到tensor时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_pretrain_emb(self, qids):
        """获取预训练嵌入"""
        device = qids.device
        batch_size, seq_len = qids.shape
        
        # 初始化嵌入列表
        content_emb = None
        analysis_emb = None
        kc_emb = None
        
        # 处理content嵌入
        if self.use_content_emb and self.content_emb_data is not None:
            # 确保嵌入在正确的设备上
            if self.content_emb_data.device != device:
                self.content_emb_data = self.content_emb_data.to(device)
            
            valid_qids = torch.clamp(qids, 0, self.content_emb_data.size(0) - 1)
            
            # 内存优化：分批处理大型嵌入索引
            if batch_size * seq_len > 10000:  # 如果序列太长，分批处理
                content_emb = torch.zeros(batch_size, seq_len, self.content_emb_data.size(1), 
                                        device=device, dtype=self.content_emb_data.dtype)
                chunk_size = 1000  # 每次处理1000个元素
                for i in range(0, batch_size * seq_len, chunk_size):
                    end_idx = min(i + chunk_size, batch_size * seq_len)
                    chunk_qids = valid_qids.view(-1)[i:end_idx]
                    chunk_emb = self.content_emb_data[chunk_qids]
                    content_emb.view(-1, self.content_emb_data.size(1))[i:end_idx] = chunk_emb
            else:
                # 使用索引操作，确保梯度能够传播
                content_emb = self.content_emb_data[valid_qids]  # (batch_size, seq_len, emb_dim)
            
            # 检查数值稳定性
            if torch.isnan(content_emb).any() or torch.isinf(content_emb).any():
                print(f"[DKT] 警告: content_emb包含NaN或Inf值")
                content_emb = torch.zeros_like(content_emb)
        
        # 处理analysis嵌入
        if self.use_analysis_emb and self.analysis_emb_data is not None:
            # 确保嵌入在正确的设备上
            if self.analysis_emb_data.device != device:
                self.analysis_emb_data = self.analysis_emb_data.to(device)
            
            valid_qids = torch.clamp(qids, 0, self.analysis_emb_data.size(0) - 1)
            
            # 内存优化：分批处理大型嵌入索引
            if batch_size * seq_len > 10000:  # 如果序列太长，分批处理
                analysis_emb = torch.zeros(batch_size, seq_len, self.analysis_emb_data.size(1), 
                                         device=device, dtype=self.analysis_emb_data.dtype)
                chunk_size = 1000  # 每次处理1000个元素
                for i in range(0, batch_size * seq_len, chunk_size):
                    end_idx = min(i + chunk_size, batch_size * seq_len)
                    chunk_qids = valid_qids.view(-1)[i:end_idx]
                    chunk_emb = self.analysis_emb_data[chunk_qids]
                    analysis_emb.view(-1, self.analysis_emb_data.size(1))[i:end_idx] = chunk_emb
            else:
                # 使用索引操作，确保梯度能够传播
                analysis_emb = self.analysis_emb_data[valid_qids]  # (batch_size, seq_len, emb_dim)
            
            # 检查数值稳定性
            if torch.isnan(analysis_emb).any() or torch.isinf(analysis_emb).any():
                print(f"[DKT] 警告: analysis_emb包含NaN或Inf值")
                analysis_emb = torch.zeros_like(analysis_emb)
        
        # 处理KC嵌入
        if self.use_kc_emb and self.kc_emb_data is not None:
            # 确保嵌入在正确的设备上
            if self.kc_emb_data.device != device:
                self.kc_emb_data = self.kc_emb_data.to(device)
            
            valid_qids = torch.clamp(qids, 0, self.kc_emb_data.size(0) - 1)
            
            # 内存优化：分批处理大型嵌入索引
            if batch_size * seq_len > 10000:  # 如果序列太长，分批处理
                kc_emb = torch.zeros(batch_size, seq_len, self.kc_emb_data.size(1), 
                                   device=device, dtype=self.kc_emb_data.dtype)
                chunk_size = 1000  # 每次处理1000个元素
                for i in range(0, batch_size * seq_len, chunk_size):
                    end_idx = min(i + chunk_size, batch_size * seq_len)
                    chunk_qids = valid_qids.view(-1)[i:end_idx]
                    chunk_emb = self.kc_emb_data[chunk_qids]
                    kc_emb.view(-1, self.kc_emb_data.size(1))[i:end_idx] = chunk_emb
            else:
                # 使用索引操作，确保梯度能够传播
                kc_emb = self.kc_emb_data[valid_qids]  # (batch_size, seq_len, emb_dim)
            
            # 检查数值稳定性
            if torch.isnan(kc_emb).any() or torch.isinf(kc_emb).any():
                print(f"[DKT] 警告: kc_emb包含NaN或Inf值")
                kc_emb = torch.zeros_like(kc_emb)
        
        # 融合嵌入
        if self.use_content_emb and self.use_analysis_emb and self.use_kc_emb and \
            content_emb is not None and analysis_emb is not None and kc_emb is not None and hasattr(self, 'cross_attention') and self.cross_attention is not None:
            try:
                enhanced_content = self.cross_attention(content_emb, analysis_emb, kc_emb)
                
                # 检查交叉注意力输出
                if torch.isnan(enhanced_content).any() or torch.isinf(enhanced_content).any():
                    print(f"[DKT] 警告: cross_attention输出包含NaN或Inf值")
                    enhanced_content = torch.zeros_like(enhanced_content)
                
                if self.no_analysis_fusion:
                    fused_emb = self.emb_fusion(enhanced_content)
                else:
                    combined_emb = torch.cat([enhanced_content, analysis_emb], dim=-1)
                    fused_emb = self.emb_fusion(combined_emb)
                
                # 检查融合输出
                if torch.isnan(fused_emb).any() or torch.isinf(fused_emb).any():
                    print(f"[DKT] 警告: emb_fusion输出包含NaN或Inf值")
                    fused_emb = torch.zeros_like(fused_emb)
                
                return fused_emb
            except Exception as e:
                print(f"[DKT] 交叉注意力计算失败: {e}")
                # 回退到简单融合
                emb_list = []
                if content_emb is not None:
                    emb_list.append(content_emb)
                if analysis_emb is not None:
                    emb_list.append(analysis_emb)
                if kc_emb is not None:
                    emb_list.append(kc_emb)
                if len(emb_list) > 0:
                    combined_emb = torch.cat(emb_list, dim=-1)
                    if self.emb_fusion is not None:
                        # 检查融合层输入维度是否匹配
                        expected_input_dim = combined_emb.shape[-1]
                        actual_input_dim = self.emb_fusion.in_features
                        if expected_input_dim != actual_input_dim:
                            print(f"[DKT] 警告: 融合层输入维度不匹配，期望: {expected_input_dim}, 实际: {actual_input_dim}")
                            # 创建临时融合层
                            temp_fusion = nn.Linear(expected_input_dim, self.emb_size).to(combined_emb.device)
                            return temp_fusion(combined_emb)
                        else:
                            return self.emb_fusion(combined_emb)
                    else:
                        return combined_emb
                return None
        else:
            emb_list = []
            if content_emb is not None:
                emb_list.append(content_emb)
            if analysis_emb is not None:
                emb_list.append(analysis_emb)
            if kc_emb is not None:
                emb_list.append(kc_emb)
            if len(emb_list) == 0:
                return None
            elif len(emb_list) == 1:
                combined_emb = emb_list[0]
            else:
                combined_emb = torch.cat(emb_list, dim=-1)
            if self.emb_fusion is not None:
                fused_emb = self.emb_fusion(combined_emb)
                return fused_emb
            else:
                return combined_emb

    def to(self, device):
        """重写to方法，确保所有嵌入数据都被移动到正确设备"""
        super().to(device)
        
        # 移动嵌入数据到正确设备
        if self.content_emb_data is not None:
            # 如果是nn.Parameter，使用to方法移动
            if isinstance(self.content_emb_data, nn.Parameter):
                self.content_emb_data.data = self.content_emb_data.data.to(device)
            else:
                self.content_emb_data = self.content_emb_data.to(device)
        if self.analysis_emb_data is not None:
            # 如果是nn.Parameter，使用to方法移动
            if isinstance(self.analysis_emb_data, nn.Parameter):
                self.analysis_emb_data.data = self.analysis_emb_data.data.to(device)
            else:
                self.analysis_emb_data = self.analysis_emb_data.to(device)
        if self.kc_emb_data is not None:
            # 如果是nn.Parameter，使用to方法移动
            if isinstance(self.kc_emb_data, nn.Parameter):
                self.kc_emb_data.data = self.kc_emb_data.data.to(device)
            else:
                self.kc_emb_data = self.kc_emb_data.to(device)
        
        return self

    def forward(self, q, r):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            # 修改为使用习题ID而不是概念ID
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)
        
        # 检查基础嵌入的数值稳定性
        if torch.isnan(xemb).any() or torch.isinf(xemb).any():
            print(f"[DKT] 警告: xemb包含NaN或Inf值")
            xemb = torch.zeros_like(xemb)
        
        # 获取预训练嵌入（如果启用）
        pretrain_emb = None
        if self.use_content_emb or self.use_analysis_emb or self.use_kc_emb:
            pretrain_emb = self.get_pretrain_emb(q)
        
        # 融合嵌入
        if pretrain_emb is not None:
            # 检查预训练嵌入的数值稳定性
            if torch.isnan(pretrain_emb).any() or torch.isinf(pretrain_emb).any():
                print(f"[DKT] 警告: pretrain_emb包含NaN或Inf值")
                pretrain_emb = torch.zeros_like(pretrain_emb)
            
            combined_features = torch.cat([xemb, pretrain_emb], dim=-1)
            h = self.fusion_layer(combined_features)
        else:
            h = xemb
        
        # 检查融合后的特征
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[DKT] 警告: 融合特征h包含NaN或Inf值")
            h = torch.zeros_like(h)
        
        # print(f"xemb.shape is {xemb.shape}")
        h, _ = self.lstm_layer(h)
        
        # 检查LSTM输出
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[DKT] 警告: LSTM输出包含NaN或Inf值")
            h = torch.zeros_like(h)
        
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        
        # 检查最终输出
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[DKT] 警告: 最终输出y包含NaN或Inf值")
            y = torch.zeros_like(y)

        # 清理中间变量以节省内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return y

# 简单的测试函数
def test_three_layer_hierarchical_attention():
    """测试三层分层注意力机制"""
    print("=== 测试三层分层注意力机制 ===")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    content_dim = 512
    analysis_dim = 1536
    kc_dim = 1600
    d_model = 128
    
    # 创建注意力机制
    attention = ThreeLayerHierarchicalAttention(
        content_dim=content_dim,
        analysis_dim=analysis_dim,
        kc_dim=kc_dim,
        d_model=d_model,
        num_heads=4,
        dropout=0.1
    )
    
    # 创建测试输入
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    analysis_emb = torch.randn(batch_size, seq_len, analysis_dim)
    kc_emb = torch.randn(batch_size, seq_len, kc_dim)
    
    print(f"输入形状:")
    print(f"  - content_emb: {content_emb.shape}")
    print(f"  - analysis_emb: {analysis_emb.shape}")
    print(f"  - kc_emb: {kc_emb.shape}")
    
    # 前向传播
    try:
        output = attention(content_emb, analysis_emb, kc_emb)
        print(f"输出形状: {output.shape}")
        print(f"输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"输出均值: {output.mean().item():.4f}")
        print(f"输出标准差: {output.std().item():.4f}")
        
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值")
        elif torch.isinf(output).any():
            print("❌ 输出包含Inf值")
        else:
            print("✅ 三层分层注意力机制测试通过")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dkt_with_three_layer_attention():
    """测试在DKT模型中使用三层分层注意力机制"""
    print("\n=== 测试DKT模型中的三层分层注意力机制 ===")
    
    try:
        # 创建DKT模型实例，使用三层分层注意力
        model = DKT(
            num_c=100,  # 概念数量
            emb_size=128,  # 嵌入维度
            dropout=0.1,
            emb_type='qid',
            use_content_emb=True,
            use_analysis_emb=True,
            use_kc_emb=True,
            content_dim=512,
            analysis_dim=1536,
            kc_dim=1600,
            attention_type="three_layer_hierarchical",  # 使用新的三层分层注意力
            dataset_name="XES3G5M",  # 指定数据集
            num_q=50  # 习题数量
        )
        
        print(f"✅ DKT模型创建成功")
        print(f"  - 注意力类型: {model.attention_type}")
        print(f"  - 嵌入维度: {model.emb_size}")
        print(f"  - 使用content嵌入: {model.use_content_emb}")
        print(f"  - 使用analysis嵌入: {model.use_analysis_emb}")
        print(f"  - 使用kc嵌入: {model.use_kc_emb}")
        
        # 创建测试输入
        batch_size = 2
        seq_len = 10
        q = torch.randint(0, 50, (batch_size, seq_len))  # 习题ID
        r = torch.randint(0, 2, (batch_size, seq_len))   # 答题结果
        
        print(f"\n测试输入:")
        print(f"  - q shape: {q.shape}")
        print(f"  - r shape: {r.shape}")
        
        # 前向传播
        output = model(q, r)
        print(f"  - 输出shape: {output.shape}")
        print(f"  - 输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值")
        elif torch.isinf(output).any():
            print("❌ 输出包含Inf值")
        else:
            print("✅ DKT模型三层分层注意力机制测试通过")
            
    except Exception as e:
        print(f"❌ DKT模型测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_three_layer_hierarchical_attention()
    test_dkt_with_three_layer_attention()