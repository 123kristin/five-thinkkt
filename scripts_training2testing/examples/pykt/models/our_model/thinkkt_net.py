"""
知识状态追踪器（Knowledge State Tracker）
用于建模学生的知识状态变化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import Optional


class CausalTransformerEncoder(nn.Module):
    """
    带因果掩码的Transformer编码器
    确保位置i只能关注到位置j<=i，防止数据泄漏
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers, activation='gelu'):
        super().__init__()
        # 创建独立的层实例
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation=activation
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        前向传播，自动添加因果掩码
        
        Args:
            src: 输入序列 (batch, seq_len, d_model)
            mask: 自定义注意力掩码（可选，会被因果掩码覆盖）
            src_key_padding_mask: padding掩码
        """
        output = src
        seq_len = src.size(1)
        
        # 创建因果掩码（上三角矩阵，True表示需要mask的位置）
        # 位置i只能关注位置j<=i
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool), diagonal=1)
        
        # 遍历每一层
        for mod in self.layers:
            output = mod(output, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        
        return output


class ThinkKTNet(nn.Module):
    """
    知识状态追踪器网络
    
    功能：
    1. 融合题目特征、答题结果、CoT嵌入、知识点分布
    2. 使用Transformer或LSTM建模时序依赖
    3. 预测答对概率
    """
    
    def __init__(self, config: dict):
        """
        初始化知识状态追踪器
        
        Args:
            config: 配置字典，包含：
                - d_question: 题目特征维度
                - d_cot: CoT嵌入维度（如果使用CoT）
                - num_c: 知识点数量
                - d_knowledge: 知识状态维度
                - seq_model_type: 序列模型类型 ("transformer" 或 "lstm")
                - num_transformer_layers: Transformer层数
                - num_heads: 注意力头数
                - dropout: Dropout率
                - use_cot: 是否使用CoT
        """
        super(ThinkKTNet, self).__init__()
        
        # 设备管理（与CRKT一致）
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # 参数设置
        self.d_question = config.get('d_question', 1024)
        self.d_cot = config.get('d_cot', 384)  # CoT嵌入维度
        self.num_c = config.get('num_c', 100)
        self.d_knowledge = config.get('d_knowledge', 512)
        self.dropout = config.get('dropout', 0.1)
        self.use_cot = config.get('use_cot', False)
        self.seq_model_type = config.get('seq_model_type', 'transformer')
        
        # 答题结果嵌入（0: 错误, 1: 正确）
        self.d_answer = self.d_question // 4  # 答题结果嵌入维度
        self.answer_emb = nn.Embedding(2, self.d_answer)
        
        # 计算融合后的输入维度
        d_input = self.d_question + self.d_answer + self.num_c
        if self.use_cot:
            d_input += self.d_cot
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_input, self.d_knowledge),
            nn.LayerNorm(self.d_knowledge),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 序列建模
        if self.seq_model_type == 'transformer':
            num_layers = config.get('num_transformer_layers', 6)
            num_heads = config.get('num_heads', 8)
            
            # 使用自定义的CausalTransformerEncoder以支持因果掩码
            self.seq_model = CausalTransformerEncoder(
                d_model=self.d_knowledge,
                nhead=num_heads,
                dim_feedforward=self.d_knowledge * 4,
                dropout=self.dropout,
                num_layers=num_layers,
                activation='gelu'
            )
        elif self.seq_model_type == 'lstm':
            num_layers = config.get('num_lstm_layers', 2)
            self.seq_model = nn.LSTM(
                input_size=self.d_knowledge,
                hidden_size=self.d_knowledge,
                num_layers=num_layers,
                batch_first=True,
                dropout=self.dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            raise ValueError(f"不支持的序列模型类型: {self.seq_model_type}")
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(self.d_knowledge, self.d_knowledge // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_knowledge // 2, 1),
            nn.Sigmoid()
        )
        
        # 知识点掌握度输出（可选，用于可解释性）
        self.kc_mastery_head = nn.Sequential(
            nn.Linear(self.d_knowledge, self.d_knowledge // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_knowledge // 2, self.num_c),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        k_t: torch.Tensor,
        r_embed: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_kc_mastery: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            v_t: 题目特征 (batch_size, seq_len, d_question)
            a_t: 答题结果 (batch_size, seq_len) 0或1
            k_t: 知识点分布 (batch_size, seq_len, num_c)
            r_embed: CoT嵌入 (batch_size, seq_len, d_cot) 或 None
            mask: 掩码 (batch_size, seq_len) True表示有效位置
            return_kc_mastery: 是否返回知识点掌握度
            
        Returns:
            y_pred: 答对概率 (batch_size, seq_len)
            或 (y_pred, kc_mastery) 如果 return_kc_mastery=True
        """
        batch_size, seq_len = v_t.shape[:2]
        device = v_t.device
        
        # 确保所有输入在正确设备上
        v_t = v_t.to(device)
        a_t = a_t.to(device)
        k_t = k_t.to(device)
        
        # 1. 答题结果嵌入
        a_emb = self.answer_emb(a_t.long())  # (batch_size, seq_len, d_answer)
        
        # 2. 特征融合
        if self.use_cot and r_embed is not None:
            r_embed = r_embed.to(device)
            z = torch.cat([v_t, a_emb, r_embed, k_t], dim=-1)  # (batch_size, seq_len, d_input)
        else:
            z = torch.cat([v_t, a_emb, k_t], dim=-1)  # (batch_size, seq_len, d_input)
        
        z = self.fusion_layer(z)  # (batch_size, seq_len, d_knowledge)
        
        # 3. 序列建模
        if self.seq_model_type == 'transformer':
            # Transformer需要处理mask
            if mask is not None:
                # mask: True表示有效位置，需要转换为False表示padding
                # Transformer期望False表示需要mask的位置
                src_key_padding_mask = ~mask.to(device)
            else:
                src_key_padding_mask = None
            
            # 使用因果掩码，确保位置i只能看到位置j<=i的信息
            # 这是通过CausalTransformerEncoderLayer自动实现的
            h_t = self.seq_model(z, src_key_padding_mask=src_key_padding_mask)
        else:  # LSTM
            # LSTM处理变长序列
            if mask is not None:
                # 使用pack_padded_sequence处理变长序列
                lengths = mask.sum(dim=1).cpu().long()  # (batch_size,)
                z_packed = nn.utils.rnn.pack_padded_sequence(
                    z, lengths, batch_first=True, enforce_sorted=False
                )
                h_t_packed, _ = self.seq_model(z_packed)
                h_t, _ = nn.utils.rnn.pad_packed_sequence(
                    h_t_packed, batch_first=True, total_length=seq_len
                )
            else:
                h_t, _ = self.seq_model(z)
        
        # 4. 预测
        y_pred = self.predictor(h_t).squeeze(-1)  # (batch_size, seq_len)
        
        if return_kc_mastery:
            kc_mastery = self.kc_mastery_head(h_t)  # (batch_size, seq_len, num_c)
            return y_pred, kc_mastery
        else:
            return y_pred
    
    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device
        return self

