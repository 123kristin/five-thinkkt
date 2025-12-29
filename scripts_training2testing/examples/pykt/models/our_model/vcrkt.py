import json
import os

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入多模态编码器相关
from .visual_language_encoder import VisualLanguageEncoder, build_img_path_dict


class VCRKTNet(nn.Module):
    def __init__(self, config: dict, data_config: dict = None):
        super(VCRKTNet, self).__init__()
        self.model_name = 'vcrkt_net'
        
        # 获取输入表征类型和维度
        # d_question_repr: 最终输入到网络的题目向量维度 (qid=200, visual=200, vq=400)
        self.d_question_repr = config.get('d_question_repr', 200) 
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            # 从环境变量获取指定的GPU ID
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[VCRKTNet] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[VCRKTNet] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[VCRKTNet] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[VCRKTNet] CUDA不可用，使用CPU")
        
        self.dropout = config.get('dropout', 0.1)

        self.num_q = config.get('num_q', 500)  # 问题数量
        self.num_c = config.get('num_c', 100)  # 知识点数量

        self.dim_qc = config.get('dim_qc', 200)  # 问题、知识点向量维度

        # 延迟创建c_ids，在to(device)时创建
        self.c_ids = None
        self.KCEmbs = nn.Embedding(self.num_c, self.dim_qc)  # 知识点嵌入

        # VCRKTNet 现在只负责接收处理好的 q_emb，不再自己管理 QEmbs 或 visual_proj
        # 但为了兼容可能得旧调用，保留 QEmbs 作为 fallback (如果 q_external_emb 为 None)
        self.QEmbs = nn.Embedding(self.num_q, self.dim_qc)
        

        # 问题难度
        self.dim_difficulty = config.get('dim_difficulty', self.dim_qc // 2)

        # 能力模块
        self.dim_knowledge = self.dim_qc
        self.rnn_type = config.get('rnn_type', 'lstm')
        
        # 计算 LSTM 输入维度: (题目 + 知识点 + 0/1 pad)
        # Shift逻辑: [q, kc, 0] vs [0, q, kc]
        # 总维度 = d_question_repr + dim_qc + (d_question_repr + dim_qc) = (d_question_repr + dim_qc) * 2
        lstm_input_dim = (self.d_question_repr + self.dim_qc) * 2
        print(f"[VCRKTNet] LSTM Input Dim: {lstm_input_dim} (Q_Repr={self.d_question_repr}, KC={self.dim_qc})")
        
        if self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(lstm_input_dim, self.dim_knowledge, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(lstm_input_dim, self.dim_knowledge, batch_first=True)

        self.q_scores_extractor = nn.Sequential(
            nn.Dropout(self.dropout),
            # nn.Linear(self.dim_knowledge, self.dim_knowledge),
            # nn.ReLU(),
            nn.Linear(self.dim_knowledge, self.num_q)
        )

    def _ensure_c_ids_on_device(self):
        """确保c_ids在正确的设备上"""
        if self.c_ids is None or self.c_ids.device != self.device:
            self.c_ids = torch.arange(self.num_c, device=self.device)

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device
        
        # 重新创建c_ids在正确设备上
        self.c_ids = torch.arange(self.num_c, device=device)
        
        return self

    def get_kc_avg_emb(self, c, pad_idx=-1):
        # 1. 掩码：True 表示有效索引
        mask = c != pad_idx  # [bz, len, max_concepts]

        # 2. 安全索引：把 -1 等填充值映射到 0（或其他任意合法 id，后面会被 mask 忽略）
        c_safe = c.masked_fill(~mask, 0)  # [bz, len, max_concepts]

        # 3. 查表得到所有向量；填充位置向量将被后续 mask 清零
        embs = self.KCEmbs(c_safe)  # [bz, len, max_concepts, emb_size]

        # 4. 将填充位置向量置 0
        embs = embs * mask.unsqueeze(-1)  # [bz, len, max_concepts, emb_size]

        # 5. 求均值（避免除 0）
        sum_emb = embs.sum(dim=-2)  # [bz, len, emb_size]
        valid_cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1) # [bz, len, 1]
        mean_emb = sum_emb / valid_cnt  # [bz, len, emb_size]

        return mean_emb

    def forward(self, q, c, r, q_shift, q_external_emb=None, return_all=False):
        """
        :param q: (bz, interactions_seq_len - 1)
            the first (interaction_seq_len - 1) q in an interaction sequence of a student
        :param c: (bz, interactions_seq_len - 1, max_concepts)
        :param r: (bz, interactions_seq_len - 1)
            the first (interaction_seq_len - 1) responses in an interaction sequence of a student
        :param q_shift: (bz, interactions_seq_len - 1)
            the last (interaction_seq_len - 1) q  in an interaction sequence of a student

        :return: (bz, interaction_seq_len - 1)
            the predicted (interaction_seq_len - 1) responses
        """
        bz, num_interactions = q.shape  # num_interactions = interactions_seq_len - 1

        # 确保c_ids在正确设备上
        self._ensure_c_ids_on_device()

        # 移入 device
        q = q.to(self.device)
        c = c.to(self.device)
        r = r.to(self.device)
        q_shift = q_shift.to(self.device)
        
        if q_external_emb is not None:
            # 优先使用外部传入的特征 (可能是 QID, Visual, 或 V&Q)
            q_emb = q_external_emb.to(self.device).float() # [bz, num_interactions, d_question_repr]
        else:
            #仅作 Fallback
            q_emb = self.QEmbs(q)  # [bz, num_interactions, dim_qc]
        
        # [bz, num_interactions, num_c]
        c_ids = self.c_ids.unsqueeze(0).expand(bz, -1).unsqueeze(1).expand(-1, num_interactions, -1)
        c_embs = self.KCEmbs(c_ids)  # (bz, num_interactions, num_c, dim_qc)

        # 构造 Zero Vector，长度需匹配 q_emb + kc_embs
        # q_emb dim: self.d_question_repr
        # kc_embs dim: self.dim_qc
        zero_dim = self.d_question_repr + self.dim_qc
        zero_vector = torch.zeros(bz, num_interactions, zero_dim, device=self.device)
        
        kc_embs = self.get_kc_avg_emb(c)  # (bz, num_interactions, dim_qc)
        
        # r 的相应位置为 1 时 q&c&0 否则 0&q&c
        # cat([q_emb, kc_embs, zero_vector]) -> length: d_q + dim_qc + (d_q + dim_qc) = 2*(d_q+dim_qc)
        e_emb = torch.where(
            r.unsqueeze(-1) == 1,  # [bz, num_interactions, 1]
            torch.cat([q_emb, kc_embs, zero_vector], dim=-1),  # rt=1 时拼接 [xt, 0]
            torch.cat([zero_vector, q_emb, kc_embs], dim=-1)  # rt=0 时拼接 [0, xt]
        )

        lstms_out, _ = self.rnn_layer(e_emb)  # (bz, interactions_seq_len - 1, dim_knowledge)
        q_scores = self.q_scores_extractor(lstms_out)  # (bz, num_interactions, num_q)
        q_scores = torch.sigmoid(q_scores) # (bz, num_interactions, num_q)
        # 计算解决该问题的能力
        y = (q_scores * F.one_hot(q_shift.long(), self.num_q)).sum(-1)  # (bz, num_interactions)

        if return_all:
            return y, q_scores, None, None
        else:
            return y


class VCRKT(nn.Module):
    """
    消融实验用，没有多头注意力的block 块，即没有建模q_kcs的权重以及
    """

    def __init__(self, config, data_config=None):
        super(VCRKT, self).__init__()
        self.model_name = 'vcrkt'
        self.emb_type = config.get('emb_type', 'qkcs')
        
        self.question_rep_type = config.get('question_rep_type', 'qid')
        self.dataset_name = config.get('dataset_name', 'DBE_KT22') 
        self.d_question = config.get('d_question', 1024)
        self.dim_qc = config.get('dim_qc', 200)
        self.num_q = config.get('num_q', 100) # 需要num_q来初始化QEmbs
        
        # 改进设备选择逻辑
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

        # --- 初始化特征提取组件 (Wrapper负责) ---
        
        # 1. QID Embedding (用于 'qid', 'vq', 'gf', 'ca', 'cl')
        if self.question_rep_type in ['qid', 'vq', 'gf', 'ca', 'cl', 'cga']:
            print(f"[VCRKT] Initializing QID Embeddings (Dim={self.dim_qc})")
            self.QEmbs = nn.Embedding(self.num_q, self.dim_qc).to(self.device)
        else:
            self.QEmbs = None
            
        # 2. Visual Projector (用于 'visual', 'vq', 'gf', 'ca', 'cl')
        if self.question_rep_type in ['visual', 'vq', 'gf', 'ca', 'cl', 'cga']:
            print(f"[VCRKT] Initializing Visual Projector ({self.d_question}->{self.dim_qc})")
            self.visual_proj = nn.Linear(self.d_question, self.dim_qc).to(self.device)
            
            # 初始化多模态编码器
            print(f"[VCRKT] Initializing Visual Encoder...")
            self.visual_encoder = VisualLanguageEncoder(
                num_c=config.get('num_c', 100),
                d_question=self.d_question, 
                model_path=config.get('mllm_name', '/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct'),
                cache_dir=config.get('cache_dir', 'features'),
                dataset_name=self.dataset_name,
                use_cache=True,
                device=self.device
            )
            
            # 构建图片路径映射
            if data_config is None:
                print(f"[VCRKT] 警告: data_config missing!")
                self.img_path_dict = {}
            else:
                self.img_path_dict = build_img_path_dict(self.dataset_name, data_config)
        else:
            self.visual_proj = None
            self.visual_encoder = None
            self.img_path_dict = {}

        # --- 初始化高级融合层 ---
        if self.question_rep_type == 'gf':
            # Gated Fusion: 学习一个门控权重
            # Input: Concat(Q, V) -> 400 dim
            # Gate: 400 -> 1 (或者 400 -> 200)
            # 这里我们用 scalar gate per channel: 400 -> 200
            print(f"[VCRKT] Initializing Gated Fusion Layer...")
            self.gate_layer = nn.Sequential(
                nn.Linear(self.dim_qc * 2, self.dim_qc),
                nn.Sigmoid()
            ).to(self.device)
            
        elif self.question_rep_type in ['ca', 'cga']:
            # Cross Attention: 
            # ca: QID as Query, Visual as Key/Value
            # cga: KC Emb as Query, Visual as Key/Value
            print(f"[VCRKT] Initializing Cross-Attention Layer (Mode={self.question_rep_type})...")
            self.attn_layer = nn.MultiheadAttention(embed_dim=self.dim_qc, num_heads=4, batch_first=True).to(self.device)
            # 残差连接通常不需要额外参数，直接相加
            
        elif self.question_rep_type == 'cl':
            # Contrastive Learning: 额外的 Loss 权重
            self.cl_weight = config.get('cl_weight', 0.1)
            print(f"[VCRKT] Contrastive Learning Mode Enabled (Weight={self.cl_weight})")

        # --- 计算传递给 VCRKTNet 的 d_question_repr ---
        if self.question_rep_type == 'qid':
            d_repr = self.dim_qc
        elif self.question_rep_type == 'visual':
            d_repr = self.dim_qc
        elif self.question_rep_type == 'vq':
            d_repr = self.dim_qc * 2 # 200 + 200 = 400
        elif self.question_rep_type == 'cl':
            d_repr = self.dim_qc * 2 # 同 VQ
        elif self.question_rep_type in ['gf', 'ca', 'cga']:
            d_repr = self.dim_qc # 融合后变回 200
        else:
            d_repr = self.dim_qc # default
            
        print(f"[VCRKT] Calculated d_question_repr for Net: {d_repr}")
        config['d_question_repr'] = d_repr
        
        self.model = VCRKTNet(config, data_config)

    def _get_question_features(self, qids, cseqs=None):
        """获取题目特征 (QID, Visual, or Combined)"""
        # qids: (batch, seq_len)
        bz, seq_len = qids.shape
        
        # 1. Get QID Emb
        v_qid = None
        if self.QEmbs is not None:
             safe_qids = qids.long().to(self.device)
             # Clamp/Handle padding if necessary (usually handled by caller or mask)
             # 这里简单处理，假设输入合法
             v_qid = self.QEmbs(safe_qids) # (bz, seq, 200)

        # 2. Get Visual Emb
        v_visual = None
        if self.visual_encoder is not None:
            # Encoder returns (bz, seq, 1024)
            v_raw, _ = self.visual_encoder(qids, self.img_path_dict, return_kc=False)
            
            # Project to 200
            if self.visual_proj is not None:
                v_visual = self.visual_proj(v_raw.to(self.device)) # (bz, seq, 200)
        
        # 3. Combine
        if self.question_rep_type == 'qid':
            return v_qid
        elif self.question_rep_type == 'visual':
            return v_visual
        elif self.question_rep_type in ['vq', 'cl']:
            if v_qid is None or v_visual is None:
                # Should not happen
                return None 
            return torch.cat([v_qid, v_visual], dim=-1) # (bz, seq, 400)
            
        elif self.question_rep_type == 'gf':
            if v_qid is None or v_visual is None: return None
            # Gated Fusion
            concat_feat = torch.cat([v_qid, v_visual], dim=-1) # (bz, seq, 400)
            gate = self.gate_layer(concat_feat) # (bz, seq, 200)
            # 融合: QID * (1 - gate) + Visual * gate (或者其他形式)
            # 这里采用残差形式: 以 QID 为主，Visual 为辅
            # 或者 symmetric
            fused = v_qid * (1 - gate) + v_visual * gate
            return fused # (bz, seq, 200)
            
        elif self.question_rep_type == 'ca':
            if v_qid is None or v_visual is None: return None
            # Cross Attention
            # Query=QID, Key=Visual, Value=Visual
            # Self-attention needs (bz, seq, embed_dim)
            attn_output, _ = self.attn_layer(query=v_qid, key=v_visual, value=v_visual)
            # Residual connection & Norm (optional, but good practice)
            # 这里简单做 Add
            return v_qid + attn_output # (bz, seq, 200)

        elif self.question_rep_type == 'cga':
            if cseqs is None or v_visual is None or v_qid is None: return v_qid
            # Concept-Guided Attention
            # Query=KC Embedding, Key=Visual, Value=Visual
            
            # 1. Get KC Avg Embedding for current step
            # cseqs: (bz, seq, max_concepts) or similar
            # Use VCRKTNet's method to get semantic representation of concept
            kc_emb = self.model.get_kc_avg_emb(cseqs.to(self.device)) # (bz, seq, 200)
            
            # 2. Attention
            attn_output, _ = self.attn_layer(query=kc_emb, key=v_visual, value=v_visual)
            
            # 3. Fuse: QID + Visual(guided by KC)
            return v_qid + attn_output # (bz, seq, 200)
            
        return v_qid

    def train_one_step(self, data):
        # 获取外部特征（如果有）
        q_external_emb = self._get_question_features(data['qseqs'], data.get('cseqs'))
        
        y = self.model(
            data['qseqs'], 
            data['cseqs'], 
            data['rseqs'], 
            data['shft_qseqs'],
            q_external_emb=q_external_emb
        )

        sm = data['smasks'].to(self.device)
        r_shift = data['shft_rseqs'].to(self.device)
        # calculate loss
        # calculate loss
        loss = self.get_loss(y, r_shift, sm)

        if self.question_rep_type == 'cl':
            # Add Contrastive Loss
            # Input q_external_emb is (bz, seq, 400)
            # Split it back
            # q_external_emb: [bz, seq, 400]
            # First 200 is QID, Last 200 is Visual
            dim_half = self.dim_qc
            v_qid = q_external_emb[:, :, :dim_half]
            v_visual = q_external_emb[:, :, dim_half:]
            
            # mask无效位置 (padding)
            # sm: [bz, seq]
            
            cl_loss = self.get_cl_loss(v_qid, v_visual, sm)
            loss = loss + self.cl_weight * cl_loss

        return y, loss

    def get_cl_loss(self, z1, z2, mask):
        """
        简单的对比损失 (Cosine Similarity)
        让同一时间步的 z1 和 z2 相似
        """
        # Flatten valid tokens
        valid_z1 = z1[mask.bool()] # [N, dim]
        valid_z2 = z2[mask.bool()] # [N, dim]
        
        if valid_z1.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
            
        # InfoNCE or simple Cosine Embedding Loss?
        # Simpler: Maximize Cosine Similarity between positive pairs
        # 1 - CosineSim
        target = torch.ones(valid_z1.size(0), device=self.device)
        loss = nn.CosineEmbeddingLoss()(valid_z1, valid_z2, target)
        return loss

    def predict_one_step(self, data):
        # 获取外部特征（如果有）
        q_external_emb = self._get_question_features(data['qseqs'], data.get('cseqs'))
        
        y = self.model(
            data['qseqs'], 
            data['cseqs'], 
            data['rseqs'], 
            data['shft_qseqs'],
             q_external_emb=q_external_emb
        )
        return y

    def get_loss(self, ys, rshft, sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        
        loss = F.binary_cross_entropy(y_pred.double(), y_true.double())
        return loss

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device
        self.model.to(device)
        return self

if __name__ == '__main__':
    config = {
        'num_q': 500,
        'num_c': 100,
        'dim_qc': 200,
        'dim_difficulty': 50,
        'dim_knowledge': 200,
        'file_map_qid2c_ids': None,
        'dropout': 0.1
    }
    model = VCRKT(config)
    print(model)
    import sys

    dir_current = os.path.dirname(__file__)
    sys.path.append(dir_current)

    bz, num_interactions = 2, 200
    q = torch.randint(0, config['num_q'], (bz, num_interactions - 1))
    r = torch.randint(0, 2, (bz, num_interactions - 1))
    q_shift = torch.randint(0, config['num_q'], (bz, num_interactions - 1))
    y = model.model(q, r, q_shift)
    print(y.shape)
