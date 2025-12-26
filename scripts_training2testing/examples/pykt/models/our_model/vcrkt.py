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
        
        # 获取输入表征类型
        self.question_rep_type = config.get('question_rep_type', 'qid')
        self.d_question = config.get('d_question', 1024) # 视觉特征维度通常是1024
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

        # 根据表征类型初始化题目部分
        if self.question_rep_type == 'visual':
            # 视觉模式：使用外部特征投影
            # 视觉特征维度(d_question) -> 目标维度(dim_qc)
            self.visual_proj = nn.Linear(self.d_question, self.dim_qc)
            # QEmbs 仍保留以防万一，或者作为 fallback，但在 forward 中不使用
            self.QEmbs = nn.Embedding(self.num_q, self.dim_qc) 
            print(f"[VCRKTNet] Visual Mode: Initialized Projection {self.d_question}->{self.dim_qc}")
        else:
            # 原始模式：使用 QID Embedding
            self.QEmbs = nn.Embedding(self.num_q, self.dim_qc)
            self.visual_proj = None
            print(f"[VCRKTNet] QID Mode: Initialized QID Embedding (dim={self.dim_qc})")

        # 问题难度
        self.dim_difficulty = config.get('dim_difficulty', self.dim_qc // 2)

        # 能力模块
        self.dim_knowledge = self.dim_qc
        self.rnn_type = config.get('rnn_type', 'lstm')
        if self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(self.dim_qc * 4, self.dim_knowledge, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(self.dim_qc * 4, self.dim_knowledge, batch_first=True)

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
        
        if self.visual_proj is not None:
            self.visual_proj = self.visual_proj.to(device)
        
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
        
        if q_external_emb is not None and self.visual_proj is not None:
            # 使用外部视觉特征
            q_external_emb = q_external_emb.to(self.device)
            q_emb = self.visual_proj(q_external_emb) # [bz, num_interactions, dim_qc]
        else:
            # 使用内部 QID Embedding
            q_emb = self.QEmbs(q)  # [bz, num_interactions, dim_qc]
        # [bz, num_interactions, num_c]
        c_ids = self.c_ids.unsqueeze(0).expand(bz, -1).unsqueeze(1).expand(-1, num_interactions, -1)
        c_embs = self.KCEmbs(c_ids)  # (bz, num_interactions, num_c, dim_qc)

        zero_vector = torch.zeros(bz, num_interactions, self.dim_qc * 2, device=self.device)
        kc_embs = self.get_kc_avg_emb(c)  # (bz, num_interactions, dim_qc)
        # r 的相应位置为 1 时 q&c&0 否则 0&q&c
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
        self.dataset_name = config.get('dataset_name', 'DBE_KT22') # 需要数据集名称来加载图片映射
        self.d_question = config.get('d_question', 1024)
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            # 从环境变量获取指定的GPU ID
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[VCRKT] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[VCRKT] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[VCRKT] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[VCRKT] CUDA不可用，使用CPU")
        
        
        # 初始化多模态组件
        if self.question_rep_type == 'visual':
            print(f"[VCRKT] 正在初始化多模态编码器 (Visual Mode)...")
            self.visual_encoder = VisualLanguageEncoder(
                num_c=config.get('num_c', 100),
                d_question=self.d_question, # Encoder输出目标维度，还是先输出1024再投影? 
                # 这里我们遵循 ThinkKT 模式：Encoder 输出缓存维度(1024) -> 投影到 d_question 
                # 但 VCRKTNet 内部有一个 d_question -> dim_qc 的投影。
                # 所以这里我们让 Encoder 输出 1024 (d_question) 即可。
                model_path=config.get('mllm_name', '/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct'),
                cache_dir=config.get('cache_dir', 'features'),
                dataset_name=self.dataset_name,
                use_cache=True,
                device=self.device
            )
            
            # 构建图片路径映射
            if data_config is None:
                print(f"[VCRKT] 警告: Visual Mode 需要 data_config 来构建图片路径映射，但未提供！")
                self.img_path_dict = {}
            else:
                self.img_path_dict = build_img_path_dict(self.dataset_name, data_config)
                print(f"[VCRKT] 已构建 {len(self.img_path_dict)} 个图片映射")
        else:
            self.visual_encoder = None
            self.img_path_dict = {}

        self.model = VCRKTNet(config, data_config)

    def _get_question_features(self, qids):
        """获取题目特征"""
        if self.question_rep_type == 'visual' and self.visual_encoder is not None:
            # qids: (batch, seq_len)
            # visual_encoder forward returns (v_t, k_t)
            # 我们只需要 v_t
            v_t, _ = self.visual_encoder(qids, self.img_path_dict, return_kc=False)
            return v_t
        return None

    def train_one_step(self, data):
        # 获取外部特征（如果有）
        q_external_emb = self._get_question_features(data['qseqs'])
        
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
        loss = self.get_loss(y, r_shift, sm)

        return y, loss

    def predict_one_step(self, data):
        # 获取外部特征（如果有）
        q_external_emb = self._get_question_features(data['qseqs'])
        
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
