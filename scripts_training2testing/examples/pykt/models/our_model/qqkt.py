import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

try:
    from .cot.cot_generator import CoTGenerator
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    CoTGenerator = None

from .visual_language_encoder import build_img_path_dict

def load_kc_vocab(dataset_name: str, data_config: dict) -> Dict[int, str]:
    """
    从数据集文件中加载知识点词表 (复用自 ThinkKT)
    """
    kc_vocab = {}
    dpath = data_config.get('dpath', '')
    
    if not dpath:
        print(f"[QQKT] 警告: data_config中没有'dpath'字段，无法加载知识点词表")
        return kc_vocab
    
    try:
        if "DBE_KT22" in dataset_name:
            # DBE_KT22: 从 KCs.csv 加载
            possible_paths = [
                os.path.join(dpath, "KCs.csv"),
                os.path.join(dpath, "../2_DBE_KT22_datafiles_100102_csv/KCs.csv"),
                os.path.join(dpath, "../../2_DBE_KT22_datafiles_100102_csv/KCs.csv"),
                "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/KCs.csv"
            ]
            kcs_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    kcs_file = path
                    break
            
            if kcs_file and os.path.exists(kcs_file):
                df_kcs = pd.read_csv(kcs_file)
                for _, row in df_kcs.iterrows():
                    kc_id = int(row['id'])
                    kc_name = str(row['name']).strip()
                    kc_vocab[kc_id] = kc_name
                print(f"[QQKT] 从 {kcs_file} 加载了 {len(kc_vocab)} 个知识点")
        
        elif "XES3G5M" in dataset_name:
            # XES3G5M: 从 metadata/kc_routes_map.json 加载
            kc_file = os.path.join(dpath, "metadata", "kc_routes_map.json")
            if not os.path.exists(kc_file):
                kc_file = os.path.join(dpath, "kc_routes_map.json")
            
            if os.path.exists(kc_file):
                with open(kc_file, 'r', encoding='utf-8') as f:
                    kc_map = json.load(f)
                for kc_id_str, kc_name in kc_map.items():
                    try:
                        kc_id = int(kc_id_str)
                        kc_vocab[kc_id] = str(kc_name).strip()
                    except (ValueError, TypeError):
                        continue
                print(f"[QQKT] 从 {kc_file} 加载了 {len(kc_vocab)} 个知识点")
        
        elif "NIPS_task34" in dataset_name or "nips_task34" in dataset_name or "Eedi" in dataset_name or "eedi" in dataset_name:
            # NIPS_task34/Eedi
            possible_paths = [
                os.path.join(dpath, "metadata", "subject_metadata.csv"),
                os.path.join(dpath, "subject_metadata.csv"),
                "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/Eedi/data/metadata/subject_metadata.csv",
                "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/NIPS_task34/metadata/subject_metadata.csv"
            ]
            subject_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    subject_file = path
                    break
            
            if subject_file and os.path.exists(subject_file):
                df_subjects = pd.read_csv(subject_file)
                for _, row in df_subjects.iterrows():
                    try:
                        subject_id = int(row['SubjectId'])
                        subject_name = str(row['Name']).strip()
                        if pd.notna(subject_id) and pd.notna(subject_name):
                            kc_vocab[subject_id] = subject_name
                    except (ValueError, TypeError):
                        continue
                print(f"[QQKT] 从 {subject_file} 加载了 {len(kc_vocab)} 个主题（作为知识点）")
                
    except Exception as e:
        print(f"[QQKT] 加载知识点词表时出错: {e}")

    return kc_vocab



class QQKTNet(nn.Module):
    def __init__(self, config: dict):
        super(QQKTNet, self).__init__()
        self.model_name = 'qqkt_net'
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            # 从环境变量获取指定的GPU ID
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[QQKTNet] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[QQKTNet] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[QQKTNet] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[QQKTNet] CUDA不可用，使用CPU")
        
        self.dropout = config.get('dropout', 0.1)

        self.num_q = config.get('num_q', 500)  # 问题数量
        self.num_c = config.get('num_c', 100)  # 知识点数量

        self.dim_qc = config.get('dim_qc', 200)  # 问题、知识点向量维度

        # 延迟创建c_ids，在to(device)时创建
        self.c_ids = None
        self.KCEmbs = nn.Embedding(self.num_c, self.dim_qc)  # 知识点嵌入
        self.QEmbs = nn.Embedding(self.num_q, self.dim_qc)  # 问题嵌入

        # 问题难度
        self.dim_difficulty = config.get('dim_difficulty', self.dim_qc // 2)

        self.use_cot = config.get('use_cot', False)
        self.d_cot = config.get('d_cot', 384)
        print(f"[QQKTNet] Use CoT: {self.use_cot}, Dim: {self.d_cot}")

        # 能力模块
        self.dim_knowledge = self.dim_qc
        self.rnn_type = config.get('rnn_type', 'lstm')
        
        # 计算 RNN 输入维度
        # 基本输入: q(dim_qc) + c(dim_qc) + zero(dim_qc) + q/zero(dim_qc) = 4 * dim_qc
        rnn_input_dim = self.dim_qc * 4
        
        if self.use_cot:
            rnn_input_dim += self.d_cot
            print(f"[QQKTNet] Adjusted RNN input dim for CoT: {rnn_input_dim}")
            
        if self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(rnn_input_dim, self.dim_knowledge, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(rnn_input_dim, self.dim_knowledge, batch_first=True)

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

    def forward(self, q, c, r, q_shift, r_embed=None, return_all=False):
        """
        :param r_embed: (bz, interactions_seq_len - 1, d_cot) - CoT embeddings
        """
        bz, num_interactions = q.shape  # num_interactions = interactions_seq_len - 1

        # 确保c_ids在正确设备上
        self._ensure_c_ids_on_device()

        # 移入 device
        q = q.to(self.device)
        c = c.to(self.device)
        r = r.to(self.device)
        q_shift = q_shift.to(self.device)
        
        if r_embed is not None:
            r_embed = r_embed.to(self.device)

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
        
        # 拼接 CoT embedding
        if self.use_cot:
            if r_embed is None:
                # 兼容性 fallback: 全0向量
                r_embed = torch.zeros(bz, num_interactions, self.d_cot, device=self.device)
            e_emb = torch.cat([e_emb, r_embed], dim=-1)

        lstms_out, _ = self.rnn_layer(e_emb)  # (bz, interactions_seq_len - 1, dim_knowledge)
        q_scores = self.q_scores_extractor(lstms_out)  # (bz, num_interactions, num_q)
        q_scores = torch.sigmoid(q_scores) # (bz, num_interactions, num_q)
        # 计算解决该问题的能力
        y = (q_scores * F.one_hot(q_shift.long(), self.num_q)).sum(-1)  # (bz, num_interactions)

        if return_all:
            return y, q_scores, None, None
        else:
            return y


class QQKT(nn.Module):
    """
    消融实验用，没有多头注意力的block 块，即没有建模q_kcs的权重以及
    """

    def __init__(self, config):
        super(QQKT, self).__init__()
        self.model_name = 'qqkt'
        self.emb_type = config.get('emb_type', 'qkcs')  # 此处不要用 pop，因为后面还要传给 model
        
        # CoT 配置
        self.use_cot = config.get('use_cot', False)
        self.d_cot = config.get('d_cot', 384)
        self.cot_threshold = config.get('cot_threshold', 2)
        self.adaptive_strategy = config.get('adaptive_strategy', 'rule')
        # 从 config 中获取 dataset_name (pykt通常会传进来)
        self.dataset_name = config.get('dataset_name', 'DBE_KT22') # Fallback default
        
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            # 从环境变量获取指定的GPU ID
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[QQKT] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[QQKT] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[QQKT] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[QQKT] CUDA不可用，使用CPU")
        
        # 初始化 CoT 生成器
        if self.use_cot:
            if not COT_AVAILABLE or CoTGenerator is None:
                print(f"[QQKT] 警告: CoT 生成器不可用，将禁用 CoT 功能")
                self.use_cot = False
                self.cot_generator = None
            else:
                print(f"[QQKT] 正在初始化 CoT 生成器...")
                sys.stdout.flush()
                self.cot_generator = CoTGenerator(
                    mllm_name=config.get('mllm_name', '/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/Qwen/Qwen2-VL-3B-Instruct'),
                    d_cot=self.d_cot,
                    cache_dir=config.get('cot_cache_dir', 'cot_cache'),
                    device=self.device,  # 共享设备
                    use_cache=True,
                    dataset_name=self.dataset_name
                )
                print(f"[QQKT] CoT 生成器初始化完成")
                sys.stdout.flush()
                
                # CoT 还需要辅助信息
                # 1. 知识点词表
                # 注意：config (model_config) 通常不包含 data_config 的全部信息，
                # 但 pykt 在 init_model 时会把 data_config 的部分信息 merge 进去或者我们假设 dpath 可能会传递
                # 这里我们假设 config 中可能没有 dpath，需要处理
                # *重要*: pykt 的 init_model 调用时，model_config 和 data_config 是分开的。
                # 但这里只传进了 config (原本是 model_config)。
                # 为了能在 QQKT 里访问 dpath，我们需要在 init_model.py 里把 dpath 塞进 model_config
                # 或者假定 data_config 已经在调用处合并了 (pykt 确实有把 num_c/num_q 塞进去)
                
                self.kc_vocab = {}
                if 'dpath' in config:
                    self.kc_vocab = load_kc_vocab(self.dataset_name, config)
                else:
                    print("[QQKT] 警告: config中缺失 'dpath'，无法加载知识点词表，CoT生成受限")
                    
                # 2. 图片路径映射
                self.img_path_dict = {}
                if 'dpath' in config:
                     self.img_path_dict = build_img_path_dict(self.dataset_name, config)
        else:
            self.cot_generator = None
            self.kc_vocab = {}
            self.img_path_dict = {}

        self.model = QQKTNet(config)


    def _get_cot_embeddings(
        self,
        qseqs: torch.Tensor,
        rseqs: torch.Tensor,
        cseqs: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        获取CoT嵌入 (复用自 ThinkKT)
        """
        if not self.use_cot or self.cot_generator is None:
            return None
        
        batch_size, seq_len = qseqs.shape
        device = qseqs.device
        
        # 使用列表收集CPU上的tensor，最后由主函数统一移动
        cot_embeds = []
        
        # 维护知识点频次 (batch_level)
        kc_counts_map = [defaultdict(int) for _ in range(batch_size)]
        
        print(f"[QQKT] Generating CoT (Strategy={self.adaptive_strategy})...")
        sys.stdout.flush()
        
        # 预先生成全0向量 (CPU)
        zero_embed = torch.zeros(self.d_cot)
        
        qseqs_list = qseqs.cpu().tolist()
        rseqs_list = rseqs.cpu().tolist()
        cseqs_list = None
        if cseqs is not None:
            cseqs_list = cseqs.cpu().tolist()
        
        processed_items = 0
        cached_count = 0
        generated_count = 0
        total_items = batch_size * seq_len
        
        for i in range(batch_size):
            history_qids = []
            history_rs = []
            history_kcs_list = [] 
            
            # 序列当前步的 CoT 列表
            seq_cot_embeds = []
            
            for j in range(seq_len):
                qid = qseqs_list[i][j]
                if qid == 0: # Padding
                    seq_cot_embeds.append(zero_embed)
                    continue
                
                if j > 0: 
                    history_qids.append(qseqs_list[i][j-1])
                    history_rs.append(rseqs_list[i][j-1])
                    if cseqs_list is not None:
                        prev_kcs = [k for k in cseqs_list[i][j-1] if k != -1]
                        history_kcs_list.append(prev_kcs)

                processed_items += 1
                
                # 获取当前题目知识点
                current_kcs = []
                if cseqs_list is not None:
                    for k in cseqs_list[i][j]:
                        if k != -1:
                            if k > 0: # 修正知识点ID，过滤掉padding
                                current_kcs.append(k)
                
                # --- 核心决策逻辑 ---
                should_generate = False
                
                # 1. 计算频次状态
                min_count = 9999
                if current_kcs:
                    for k in current_kcs:
                        c = kc_counts_map[i][k]
                        if c < min_count: min_count = c
                else: 
                    min_count = 0
                
                # 2. 策略分支
                if self.adaptive_strategy == 'rule':
                    if min_count < self.cot_threshold:
                        should_generate = True
                
                # 获取当前题目图片路径
                img_path = None
                if qid in self.img_path_dict:
                    img_path = self.img_path_dict[qid]
                
                # 若无图片且必须用图片生成(取决于CoTGenerator实现)，则可能跳过或仅用文本
                # 这里假设CoTGenerator能处理img_path=None的情况(纯文本CoT)
                
                # 生成 CoT
                if should_generate:
                    try:
                        cache_key = self.cot_generator._get_cache_key(history_qids, history_rs, qid)
                        from_cache = cache_key in self.cot_generator.cot_cache
                        
                        _, cot_embed = self.cot_generator.generate_cot(
                            history_qids=history_qids,
                            history_rs=history_rs,
                            current_qid=qid,
                            img_path=img_path,
                            kc_vocab=self.kc_vocab,
                            history_kcs=history_kcs_list,
                            current_kcs=current_kcs
                        )
                        
                        if from_cache:
                            cached_count += 1
                        else:
                            generated_count += 1
                        
                        seq_cot_embeds.append(cot_embed.cpu())
                        
                        if processed_items % 10 == 0 or not from_cache:
                            print(f"[QQKT] CoT进度: {processed_items}/{total_items} ({100*processed_items/total_items:.1f}%) | 缓存:{cached_count} 生成:{generated_count} | qid={qid}", end='\r')
                            sys.stdout.flush()
                            
                    except Exception as e:
                        print(f"\n[QQKT] 警告: 生成 CoT 失败: {e}")
                        seq_cot_embeds.append(zero_embed)
                else:
                    seq_cot_embeds.append(zero_embed)
                
                # 更新知识点频次
                if current_kcs:
                    for k in current_kcs:
                        kc_counts_map[i][k] += 1
            
            cot_embeds.append(torch.stack(seq_cot_embeds).to(device))
        
        print(f"\n[QQKT] CoT生成完成: 总计 {processed_items} 个，缓存: {cached_count}，新生成: {generated_count}")
        sys.stdout.flush()
        
        r_embed = torch.stack(cot_embeds)
        return r_embed

    def train_one_step(self, data):
        # 获取知识点序列 (可能不在 data 中，取决于 DataLoader)
        cseqs = data.get('cseqs', None)
        # 获取 CoT 嵌入
        r_embed = self._get_cot_embeddings(data['qseqs'], data['rseqs'], cseqs)
        
        y = self.model(data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'], r_embed)

        sm = data['smasks'].to(self.device)
        r_shift = data['shft_rseqs'].to(self.device)
        # calculate loss
        loss = self.get_loss(y, r_shift, sm)

        return y, loss

    def predict_one_step(self, data):
        # 获取知识点序列
        cseqs = data.get('cseqs', None)
        # 获取 CoT 嵌入
        r_embed = self._get_cot_embeddings(data['qseqs'], data['rseqs'], cseqs)
        
        y = self.model(data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'], r_embed)
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
    model = QQKT(config)
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
