import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import math

class TripleModalFusion(nn.Module):
    """
    三模态融合注意力：
    - Content作为Query
    - Analysis和KC作为Key和Value
    - 使用门控机制控制不同模态的重要性
    """
    def __init__(self, content_dim, analysis_dim, kc_dim, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 投影层
        self.content_proj = nn.Linear(content_dim, d_model)
        self.analysis_proj = nn.Linear(analysis_dim, d_model)
        self.kc_proj = nn.Linear(kc_dim, d_model)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 门控机制
        self.gate_analysis = nn.Linear(d_model, 1)
        self.gate_kc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 融合层
        self.fusion_layer = nn.Linear(d_model * 3, d_model)
        
    def forward(self, content_emb, analysis_emb, kc_emb):
        # 投影到统一维度
        content_proj = self.content_proj(content_emb)
        analysis_proj = self.analysis_proj(analysis_emb)
        kc_proj = self.kc_proj(kc_emb)
        
        # 计算门控权重
        gate_analysis = self.sigmoid(self.gate_analysis(content_proj))
        gate_kc = self.sigmoid(self.gate_kc(content_proj))
        
        # 加权融合Key和Value
        weighted_analysis = gate_analysis * analysis_proj
        weighted_kc = gate_kc * kc_proj
        
        # 拼接作为Key和Value
        key_value = torch.cat([weighted_analysis, weighted_kc], dim=-1)
        key_value_proj = self.fusion_layer(key_value)
        
        # 交叉注意力
        attended, _ = self.multihead_attn(
            content_proj, key_value_proj, key_value_proj
        )
        
        return attended

class HierarchicalFusionAttention(nn.Module):
    """
    三层融合注意力机制：
    1. 第一层：Content-Analysis-KC三模态融合
    2. 第二层：时序自注意力建模
    3. 第三层：因果自注意力建模
    """
    def __init__(self, content_dim=512, analysis_dim=1536, kc_dim=1600, 
                 d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 第一层：三模态融合注意力
        self.triple_fusion = TripleModalFusion(
            content_dim, analysis_dim, kc_dim, d_model, num_heads, dropout
        )
        
        # 第二层：时序自注意力
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 第三层：因果自注意力
        self.causal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, content_emb, analysis_emb, kc_emb):
        """
        Args:
            content_emb: [batch_size, seq_len, content_dim]
            analysis_emb: [batch_size, seq_len, analysis_dim]
            kc_emb: [batch_size, seq_len, kc_dim]
        """
        # 第一层：三模态融合
        fused = self.triple_fusion(content_emb, analysis_emb, kc_emb)
        fused = self.norm1(fused)
        
        # 第二层：时序自注意力
        temporal, _ = self.temporal_attention(fused, fused, fused)
        temporal = self.norm2(fused + temporal)
        temporal = temporal + self.ffn1(temporal)
        
        # 第三层：因果自注意力（使用因果掩码）
        causal_mask = self._create_causal_mask(temporal.size(1), temporal.device)
        causal, _ = self.causal_attention(
            temporal, temporal, temporal, 
            attn_mask=causal_mask
        )
        output = self.norm3(temporal + causal)
        output = output + self.ffn2(output)
        
        return output
    
    def _create_causal_mask(self, seq_len, device):
        """创建因果掩码，确保只能看到历史信息"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(device)

class ImprovedDKT(Module):
    """
    改进的DKT模型，集成三模态融合注意力机制
    支持Content、Analysis和KC嵌入的融合
    """
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,
                 use_content_emb=False, use_analysis_emb=False, use_kc_emb=False, gen_emb_path="", 
                 content_dim=512, analysis_dim=1536, kc_dim=1600, analysis_type="generated",
                 trainable_content_emb=False, trainable_analysis_emb=False, trainable_kc_emb=False, 
                 num_q=0, content_type="text", **kwargs):
        super().__init__()
        self.model_name = "improved_dkt"
        self.num_c = num_c
        self.num_q = num_q
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
        self.kc_dim = kc_dim
        self.analysis_type = analysis_type
        self.content_type = content_type
        self.trainable_content_emb = trainable_content_emb
        self.trainable_analysis_emb = trainable_analysis_emb
        self.trainable_kc_emb = trainable_kc_emb
        
        # 嵌入文件路径映射
        self.emb_file_mapping = {
            "content": "embedding_content.pkl",
            "content_image": "embedding_images_content.pkl",
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
            self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)

        # 改进的注意力融合模块
        if self.use_content_emb and self.use_analysis_emb and self.use_kc_emb:
            self.hierarchical_attention = HierarchicalFusionAttention(
                content_dim=self.content_dim,
                analysis_dim=self.analysis_dim,
                kc_dim=self.kc_dim,
                d_model=self.emb_size,
                num_heads=8,
                dropout=dropout
            )
        
        # 嵌入融合层
        if self.use_content_emb or self.use_analysis_emb or self.use_kc_emb:
            self.fusion_layer = nn.Linear(self.emb_size * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        
        # 加载嵌入数据
        if gen_emb_path and (use_content_emb or use_analysis_emb or use_kc_emb):
            self._load_embeddings()

    def _load_embeddings(self):
        """加载预训练嵌入数据"""
        try:
            if self.use_content_emb:
                self._load_content_embedding()
            
            if self.use_analysis_emb:
                self._load_analysis_embedding()
            
            if self.use_kc_emb:
                self._load_kc_embedding()
                    
        except Exception as e:
            print(f"[ImprovedDKT] 加载嵌入异常: {e}")
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
                print(f"[ImprovedDKT] 不支持的内容嵌入类型: {self.content_type}")
                self.content_emb_data = None
                return
                
            pkl_path = os.path.join(self.gen_emb_path, self.emb_file_mapping[content_key])
            print(f"[ImprovedDKT] 尝试加载{self.content_type}内容嵌入文件: {pkl_path}")
            
            if os.path.exists(pkl_path):
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    print(f"[ImprovedDKT] 成功加载pkl文件，嵌入字典大小: {len(emb_dict)}")
                    self.content_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_content_emb
                    )
                    if self.content_emb_data is not None:
                        actual_content_dim = self.content_emb_data.shape[1]
                        if actual_content_dim != self.content_dim:
                            print(f"[ImprovedDKT] 自动调整content_dim: {self.content_dim} -> {actual_content_dim}")
                            self.content_dim = actual_content_dim
                        print(f"[ImprovedDKT] 成功加载{self.content_type}内容嵌入，shape: {self.content_emb_data.shape}")
                        return
                    else:
                        print(f"[ImprovedDKT] 转换嵌入字典到tensor失败")
                else:
                    print(f"[ImprovedDKT] 加载pkl文件失败或返回None")
            else:
                print(f"[ImprovedDKT] 嵌入文件不存在: {pkl_path}")
            
            print(f"[ImprovedDKT] 未找到{self.content_type}内容嵌入文件: {pkl_path}")
            self.content_emb_data = None
            
        except Exception as e:
            print(f"[ImprovedDKT] 加载{self.content_type}内容嵌入时发生异常: {e}")
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
                self.analysis_emb_data = self._convert_dict_to_tensor(
                    emb_dict, make_trainable=self.trainable_analysis_emb
                )
                if self.analysis_emb_data is not None:
                    actual_analysis_dim = self.analysis_emb_data.shape[1]
                    if actual_analysis_dim != self.analysis_dim:
                        self.analysis_dim = actual_analysis_dim
                    print(f"[ImprovedDKT] 成功加载analysis嵌入，shape: {self.analysis_emb_data.shape}")
                    return
                else:
                    print(f"[ImprovedDKT] 转换analysis嵌入字典到tensor失败")
        
        print(f"[ImprovedDKT] 未找到analysis嵌入文件: {pkl_path}")
        self.analysis_emb_data = None
    
    def _load_kc_embedding(self):
        """加载KC嵌入"""
        # 优先尝试graph_embedding_kc.pkl，如果不存在则使用embedding_kc.pkl
        kc_files = ["kc_graph", "kc"]
        for kc_key in kc_files:
            pkl_path = os.path.join(self.gen_emb_path, self.emb_file_mapping[kc_key])
            if os.path.exists(pkl_path):
                print(f"[ImprovedDKT] 尝试加载KC嵌入文件: {pkl_path}")
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    self.kc_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_kc_emb
                    )
                    if self.kc_emb_data is not None:
                        actual_kc_dim = self.kc_emb_data.shape[1]
                        if actual_kc_dim != self.kc_dim:
                            print(f"[ImprovedDKT] 自动调整kc_dim: {self.kc_dim} -> {actual_kc_dim}")
                            self.kc_dim = actual_kc_dim
                        print(f"[ImprovedDKT] 成功加载KC嵌入，shape: {self.kc_emb_data.shape}")
                        return
                    else:
                        print(f"[ImprovedDKT] 转换KC嵌入字典到tensor失败")
                else:
                    print(f"[ImprovedDKT] 加载KC嵌入pkl文件失败或返回None")
            else:
                print(f"[ImprovedDKT] KC嵌入文件不存在: {pkl_path}")
        
        print(f"[ImprovedDKT] 未找到KC嵌入文件")
        self.kc_emb_data = None
    
    def _load_pkl_embedding(self, file_path):
        """加载pkl格式嵌入文件"""
        try:
            with open(file_path, 'rb') as f:
                emb_dict = pickle.load(f)
            return emb_dict
        except Exception as e:
            print(f"[ImprovedDKT] 加载pkl文件失败 {file_path}: {e}")
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
                    print(f"[ImprovedDKT] 无法将max_qid转换为整数: {max_qid}")
                    return None
            
            # 获取嵌入维度
            sample_emb = next(iter(emb_dict.values()))
            if isinstance(sample_emb, list):
                emb_dim = len(sample_emb)
            elif hasattr(sample_emb, 'shape'):
                emb_dim = sample_emb.shape[-1]
            else:
                emb_dim = len(sample_emb)
            
            print(f"[ImprovedDKT] 嵌入维度: {emb_dim}, 最大qid: {max_qid} (类型: {type(max_qid)})")
            
            # 创建tensor并移动到正确设备
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
                        print(f"[ImprovedDKT] qid {qid} 超出tensor范围 {tensor_size}，跳过")
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
                    print(f"[ImprovedDKT] 处理嵌入qid={qid}时出错: {e}, 嵌入类型: {type(emb)}")
                    skipped_count += 1
                    continue
            
            print(f"[ImprovedDKT] 成功加载 {loaded_count} 个嵌入，跳过 {skipped_count} 个嵌入")
            
            # 如果指定为可训练，则转换为nn.Parameter
            if make_trainable:
                return nn.Parameter(emb_tensor, requires_grad=True)
            else:
                return emb_tensor
                
        except Exception as e:
            print(f"[ImprovedDKT] 转换嵌入字典到tensor时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_pretrain_emb(self, qids):
        """获取预训练嵌入，使用改进的注意力机制"""
        device = qids.device
        batch_size, seq_len = qids.shape
        
        # 获取各种嵌入
        content_emb = self._get_embedding(self.content_emb_data, qids, device)
        analysis_emb = self._get_embedding(self.analysis_emb_data, qids, device)
        kc_emb = self._get_embedding(self.kc_emb_data, qids, device)
        
        # 使用改进的注意力机制
        if (content_emb is not None and analysis_emb is not None and kc_emb is not None 
            and hasattr(self, 'hierarchical_attention')):
            enhanced_emb = self.hierarchical_attention(content_emb, analysis_emb, kc_emb)
            return enhanced_emb
        else:
            # 回退到简单的融合策略
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
                return emb_list[0]
            else:
                combined_emb = torch.cat(emb_list, dim=-1)
                # 简单的线性投影
                projection = nn.Linear(combined_emb.shape[-1], self.emb_size).to(device)
                return projection(combined_emb)
    
    def _get_embedding(self, emb_data, qids, device):
        """安全地获取嵌入"""
        if emb_data is not None:
            if emb_data.device != device:
                emb_data = emb_data.to(device)
            valid_qids = torch.clamp(qids, 0, emb_data.size(0) - 1)
            return emb_data[valid_qids]
        return None

    def to(self, device):
        """重写to方法，确保所有嵌入数据都被移动到正确设备"""
        super().to(device)
        
        # 移动嵌入数据到正确设备
        if self.content_emb_data is not None:
            if isinstance(self.content_emb_data, nn.Parameter):
                self.content_emb_data.data = self.content_emb_data.data.to(device)
            else:
                self.content_emb_data = self.content_emb_data.to(device)
        if self.analysis_emb_data is not None:
            if isinstance(self.analysis_emb_data, nn.Parameter):
                self.analysis_emb_data.data = self.analysis_emb_data.data.to(device)
            else:
                self.analysis_emb_data = self.analysis_emb_data.to(device)
        if self.kc_emb_data is not None:
            if isinstance(self.kc_emb_data, nn.Parameter):
                self.kc_emb_data.data = self.kc_emb_data.data.to(device)
            else:
                self.kc_emb_data = self.kc_emb_data.to(device)
        
        return self

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)
        
        # 获取预训练嵌入（如果启用）
        pretrain_emb = None
        if self.use_content_emb or self.use_analysis_emb or self.use_kc_emb:
            pretrain_emb = self.get_pretrain_emb(q)
        
        # 融合嵌入
        if pretrain_emb is not None:
            combined_features = torch.cat([xemb, pretrain_emb], dim=-1)
            h = self.fusion_layer(combined_features)
        else:
            h = xemb
        
        h, _ = self.lstm_layer(h)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y 