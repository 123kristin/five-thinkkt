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
        
        # 优先从 data_config 读取真实的数据集统计信息 (Crucial for XES/NIPS)
        self.num_c = data_config.get('num_c') or config.get('num_c', 100)
        self.num_q = data_config.get('num_q') or config.get('num_q', 500)
        self.d_knowledge = config.get('d_knowledge', 512)
        self.dropout = config.get('dropout', 0.1)
        self.use_cot = config.get('use_cot', False)
        self.seq_model_type = config.get('seq_model_type', 'lstm') # Default to LSTM to match CRKT
        self.prediction_mode = config.get('prediction_mode', 'vector') # Default to Vector to match CRKT
        
        # 知识点嵌入维度（与题目特征维度一致，用于知识点平均嵌入）
        self.dim_qc = self.d_question  # 知识点嵌入维度
        self.KCEmbs = nn.Embedding(self.num_c, self.dim_qc)  # 知识点嵌入
        
        # 计算融合后的输入维度
        # 不使用CoT: v_t (d_question) + kc_avg_embs (dim_qc) + zero_vector (dim_qc * 2)
        # 使用CoT: v_t (d_question) + r_embed (d_cot) + kc_avg_embs (dim_qc) + zero_vector (dim_qc * 2)
        d_input = self.d_question + self.dim_qc + self.dim_qc * 2
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
        # 预测头
        if self.prediction_mode == 'vector':
            # CRKT风格：全域预测 (Vector Output)
            # 结构：Dropout -> Linear(dim, num_q) (参考 crkt.py)
            self.predictor = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_knowledge, config.get('num_q', 500)) # 使用传入的num_q
            )
        else:
            # ThinkKT风格：单点预测 (Scalar Output)
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
    
    def get_kc_avg_emb(self, c, pad_idx=-1):
        """
        计算知识点平均嵌入（类似CRKT）
        
        Args:
            c: 知识点序列 (batch_size, seq_len, max_concepts)
            pad_idx: 填充索引，默认为-1
            
        Returns:
            mean_emb: 知识点平均嵌入 (batch_size, seq_len, dim_qc)
        """
        # 1. 掩码：True 表示有效索引
        mask = c != pad_idx  # [bz, len, max_concepts]
        
        # 2. 安全索引：把 -1 等填充值映射到 0（后面会被 mask 忽略）
        c_safe = c.masked_fill(~mask, 0)  # [bz, len, max_concepts]
        
        # 3. 查表得到所有向量；填充位置向量将被后续 mask 清零
        embs = self.KCEmbs(c_safe)  # [bz, len, max_concepts, dim_qc]
        
        # 4. 将填充位置向量置 0
        embs = embs * mask.unsqueeze(-1)  # [bz, len, max_concepts, dim_qc]
        
        # 5. 求均值（避免除 0）
        sum_emb = embs.sum(dim=-2)  # [bz, len, dim_qc]
        valid_cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [bz, len, 1]
        mean_emb = sum_emb / valid_cnt  # [bz, len, dim_qc]
        
        return mean_emb
    
    def forward(
        self,
        v_t: torch.Tensor,
        c: torch.Tensor,
        r: torch.Tensor,
        r_embed: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        q_shift: Optional[torch.Tensor] = None, # 新增：用于 Vector 模式下的选择
        return_kc_mastery: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            v_t: 题目特征 (batch_size, seq_len, d_question)
            c: 知识点序列 (batch_size, seq_len, max_concepts)
            r: 答题结果 (batch_size, seq_len) 0或1
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
        c = c.to(device)
        r = r.to(device)
        
        # 1. 计算知识点平均嵌入
        kc_avg_embs = self.get_kc_avg_emb(c)  # (batch_size, seq_len, dim_qc)
        
        # 2. 创建零向量
        zero_vector = torch.zeros(batch_size, seq_len, self.dim_qc * 2, device=device)
        
        # 3. 根据答对/答错拼接输入（类似CRKT）
        # 答对时（r==1）：[v_t, kc_avg_embs, zero_vector]
        # 答错时（r==0）：[zero_vector, v_t, kc_avg_embs]
        if self.use_cot and r_embed is not None:
            r_embed = r_embed.to(device)
            # 如果有CoT，将其插入到中间位置
            # 答对：[v_t, r_embed, kc_avg_embs, zero_vector]
            # 答错：[zero_vector, v_t, r_embed, kc_avg_embs]
            correct_input = torch.cat([v_t, r_embed, kc_avg_embs, zero_vector], dim=-1)
            wrong_input = torch.cat([zero_vector, v_t, r_embed, kc_avg_embs], dim=-1)
        else:
            correct_input = torch.cat([v_t, kc_avg_embs, zero_vector], dim=-1)
            wrong_input = torch.cat([zero_vector, v_t, kc_avg_embs], dim=-1)
        
        # 使用掩码选择答对/答错的输入
        mask_r = (r == 1).unsqueeze(-1).expand_as(correct_input)  # (batch_size, seq_len, d_input)
        z = torch.where(mask_r, correct_input, wrong_input)  # (batch_size, seq_len, d_input)
        
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
        # 4. 预测
        if self.prediction_mode == 'vector':
            # vector模式：预测所有题目的得分，然后根据q_shift选择
            q_scores = self.predictor(h_t) # (batch, seq, num_q)
            q_scores = torch.sigmoid(q_scores)
            
            if q_shift is not None:
                # 类似 CRKT 的 gather 操作
                # q_shift: (batch, seq)
                # F.one_hot dim needs to match q_scores last dim
                # output: (batch, seq)
                num_q = q_scores.size(-1)
                
                # 处理 Padding: 将负数索引 (如 -1) 替换为 0，防止 one_hot 报错
                # 这些位置在 loss 计算时会被 mask 掉，所以预测值无所谓
                safe_q_shift = q_shift.long()
                safe_q_shift = torch.where(safe_q_shift >= 0, safe_q_shift, torch.zeros_like(safe_q_shift))
                
                y_pred = (q_scores * F.one_hot(safe_q_shift, num_q)).sum(-1)
            else:
                # 如果没有提供 q_shift（例如推理时？），暂时不支持或默认行为
                # 或者返回 max/mean? 这里为了安全抛出错误或返回 scalar 形式的所有
                raise ValueError("In 'vector' prediction_mode, q_shift must be provided to forward().")
        else:
            # scalar模式
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
        # 确保知识点嵌入也在正确设备上
        self.KCEmbs = self.KCEmbs.to(device)
        return self

