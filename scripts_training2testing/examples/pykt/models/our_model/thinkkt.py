"""
ThinkKT 主模型
多模态知识追踪模型，结合视觉特征和思维链推理
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
import json
import pandas as pd

# 导入自定义模块
from .visual_language_encoder import VisualLanguageEncoder, build_img_path_dict
from .thinkkt_net import ThinkKTNet
try:
    from .cot.cot_generator import CoTGenerator
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    CoTGenerator = None


def load_kc_vocab(dataset_name: str, data_config: dict) -> Dict[int, str]:
    """
    从数据集文件中加载知识点词表
    
    Args:
        dataset_name: 数据集名称
        data_config: 数据配置字典，需要包含 'dpath' 字段
        
    Returns:
        kc_vocab: 知识点词表 {kc_id: kc_name}
    """
    kc_vocab = {}
    dpath = data_config.get('dpath', '')
    
    if not dpath:
        print(f"[ThinkKT] 警告: data_config中没有'dpath'字段，无法加载知识点词表")
        return kc_vocab
    
    try:
        if "DBE_KT22" in dataset_name:
            # DBE_KT22: 从 KCs.csv 加载
            # 格式: id,name,description
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(dpath, "KCs.csv"),
                os.path.join(dpath, "../2_DBE_KT22_datafiles_100102_csv/KCs.csv"),
                os.path.join(dpath, "../../2_DBE_KT22_datafiles_100102_csv/KCs.csv"),
                "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/KCs.csv",
                "/home3/zhiyu/code-4/kt_analysis_generation/data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/KCs.csv"
            ]
            
            kcs_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    kcs_file = path
                    break
            
            if kcs_file and os.path.exists(kcs_file):
                df_kcs = pd.read_csv(kcs_file)
                # 确保 id 是整数类型
                for _, row in df_kcs.iterrows():
                    kc_id = int(row['id'])
                    kc_name = str(row['name']).strip()
                    kc_vocab[kc_id] = kc_name
                print(f"[ThinkKT] 从 {kcs_file} 加载了 {len(kc_vocab)} 个知识点")
            else:
                print(f"[ThinkKT] 警告: 找不到KCs.csv文件，尝试过的路径: {possible_paths[:3]}")
        
        elif "XES3G5M" in dataset_name:
            # XES3G5M: 从 metadata/kc_routes_map.json 加载
            # 格式: {"0": "知识点名称", "1": "知识点名称", ...}
            kc_file = os.path.join(dpath, "metadata", "kc_routes_map.json")
            if not os.path.exists(kc_file):
                # 尝试其他可能的路径
                kc_file = os.path.join(dpath, "kc_routes_map.json")
            
            if os.path.exists(kc_file):
                with open(kc_file, 'r', encoding='utf-8') as f:
                    kc_map = json.load(f)
                # 将字符串键转换为整数
                for kc_id_str, kc_name in kc_map.items():
                    try:
                        kc_id = int(kc_id_str)
                        kc_vocab[kc_id] = str(kc_name).strip()
                    except (ValueError, TypeError):
                        continue
                print(f"[ThinkKT] 从 {kc_file} 加载了 {len(kc_vocab)} 个知识点")
            else:
                print(f"[ThinkKT] 警告: 找不到文件 {kc_file}")
        
        elif "NIPS_task34" in dataset_name or "nips_task34" in dataset_name or "Eedi" in dataset_name or "eedi" in dataset_name:
            # NIPS_task34/Eedi: 从 metadata/subject_metadata.csv 加载
            # 格式: SubjectId,Name,ParentId,Level
            # 注意：NIPS_task34 和 Eedi 是同一个数据集
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(dpath, "metadata", "subject_metadata.csv"),
                os.path.join(dpath, "../metadata/subject_metadata.csv"),
                os.path.join(dpath, "../../metadata/subject_metadata.csv"),
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
                # 确保 SubjectId 是整数类型
                for _, row in df_subjects.iterrows():
                    try:
                        subject_id = int(row['SubjectId'])
                        subject_name = str(row['Name']).strip()
                        # 跳过空值
                        if pd.notna(subject_id) and pd.notna(subject_name):
                            kc_vocab[subject_id] = subject_name
                    except (ValueError, TypeError):
                        continue
                print(f"[ThinkKT] 从 {subject_file} 加载了 {len(kc_vocab)} 个主题（作为知识点）")
            else:
                print(f"[ThinkKT] 警告: 找不到subject_metadata.csv文件，尝试过的路径: {possible_paths[:3]}")
        
        else:
            print(f"[ThinkKT] 警告: 未知的数据集 {dataset_name}，无法加载知识点词表")
    
    except Exception as e:
        print(f"[ThinkKT] 加载知识点词表时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return kc_vocab


class ThinkKT(nn.Module):
    """
    ThinkKT 模型
    
    实现 pykt 标准接口：
    - train_one_step(data) -> (y, loss)
    - predict_one_step(data) -> y
    """
    
    def __init__(self, config: dict, data_config: dict, emb_type: str = 'qkcs'):
        """
        初始化 ThinkKT 模型
        
        Args:
            config: 模型配置字典
            data_config: 数据配置字典
            emb_type: 嵌入类型（兼容pykt接口）
        """
        super(ThinkKT, self).__init__()
        
        self.model_name = 'thinkkt'
        self.emb_type = emb_type
        
        # 题目表征来源
        self.question_rep_type = config.get('question_rep_type', 'visual') # 'visual' or 'qid'
        # 优先读取 data_config，确保在创建 Embedding 前拿到正确维度
        self.num_q = data_config.get('num_q') or config.get('num_q', 500)
        self.num_c = data_config.get('num_c') or config.get('num_c', 100)
        print(f"[ThinkKT] DEBUG: Initialized ThinkKT with num_q={self.num_q}")
        self.d_question = config.get('d_question', 1024)

        if self.question_rep_type in ['qid', 'v&q']:
            # CRKT Default Dim = 200
            self.qid_dim = 200 
            print(f"[ThinkKT] Initializing QID Embeddings (Dim={self.qid_dim}) for mode: {self.question_rep_type}")
            self.QEmbs = nn.Embedding(self.num_q, self.qid_dim)
        else:
            self.QEmbs = None
            
        if self.question_rep_type in ['visual', 'v&q']:
            # Visual Projector: 1024 -> 200
            self.visual_dim = 1024 # Default visual dim from Qwen
            self.proj_dim = 200    # Target projection dim matching CRKT
            print(f"[ThinkKT] Initializing Visual Projector ({self.visual_dim} -> {self.proj_dim}) for mode: {self.question_rep_type}")
            self.visual_projector = nn.Linear(self.visual_dim, self.proj_dim)
        else:
            self.visual_projector = None
            
        # Determine effective d_question for inner network
        if self.question_rep_type == 'qid':
            self.net_d_question = 200
        elif self.question_rep_type == 'visual':
            self.net_d_question = 200 # Projected
        elif self.question_rep_type == 'v&q':
            self.net_d_question = 400 # 200 (QID) + 200 (Visual Proj)
        else:
            self.net_d_question = self.d_question # Fallback
            
        # Override config for ThinkKTNet
        print(f"[ThinkKT] Overriding d_question for Inner Net: {self.net_d_question}")
        config['d_question'] = self.net_d_question
        
        # 设备管理（与CRKT一致）
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[ThinkKT] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[ThinkKT] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[ThinkKT] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[ThinkKT] CUDA不可用，使用CPU")
        
        # 从data_config获取数据集信息
        self.num_q = data_config.get('num_q', 500)
        self.num_c = data_config.get('num_c', 100)
        self.dataset_name = config.get('dataset_name', 'DBE_KT22')
        
        # 模型配置
        self.d_question = config.get('d_question', 1024)
        self.d_cot = config.get('d_cot', 384)
        self.use_cot = config.get('use_cot', False)  # 初始版本先不使用CoT
        self.use_visual = config.get('use_visual', True)
        self.dataset_name = config.get('dataset_name', data_config.get('dpath', '').split('/')[-1]) # 优先从config获取准确的dataset_name
        
        # CoT配置
        # --- 新增配置 ---
        self.cot_threshold = config.get('cot_threshold', 2) # 稀疏策略阈值
        self.adaptive_strategy = config.get('adaptive_strategy', 'rule') # 策略模式: 'rule' (规则) 或 'learnable' (RL网络)
        print(f"[ThinkKT] Adaptive Strategy: {self.adaptive_strategy}")
        if self.adaptive_strategy == 'rule':
            print(f"[ThinkKT] Using Rule-based Threshold: {self.cot_threshold}")
        # ----------------
        
        # 初始化多模态编码器
        if self.use_visual:
            print(f"[ThinkKT] 正在初始化多模态编码器...")
            self.visual_encoder = VisualLanguageEncoder(
                num_c=self.num_c,
                d_question=self.d_question,
                model_path=config.get('mllm_name', '/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct'),
                cache_dir=config.get('cache_dir', 'features'),
                dataset_name=self.dataset_name,
                use_cache=True,
                device=self.device
            )
            
            # 构建问题ID到图片路径的映射
            print(f"[ThinkKT] 正在构建图片路径映射...")
            self.img_path_dict = build_img_path_dict(self.dataset_name, data_config)
            print(f"[ThinkKT] 已构建 {len(self.img_path_dict)} 个问题ID到图片路径的映射")
        else:
            self.visual_encoder = None
            self.img_path_dict = {}
        
        # 初始化 CoT 生成器
        if self.use_cot:
            if not COT_AVAILABLE or CoTGenerator is None:
                print(f"[ThinkKT] 警告: CoT 生成器不可用，将禁用 CoT 功能")
                self.use_cot = False
                self.cot_generator = None
            else:
                print(f"[ThinkKT] 正在初始化 CoT 生成器...")
                import sys
                sys.stdout.flush()
                self.cot_generator = CoTGenerator(
                    mllm_name=config.get('mllm_name', '/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct'),
                    text_encoder_name=config.get('text_encoder_name', '/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
                    d_cot=self.d_cot,
                    cache_dir=config.get('cot_cache_dir', 'cot_cache'),
                    device=self.device,
                    use_cache=True,
                    dataset_name=self.dataset_name
                )
                print(f"[ThinkKT] CoT 生成器初始化完成")
                sys.stdout.flush()
        else:
            self.cot_generator = None
        
        # 构建知识点词表（从数据集文件中加载）
        self.kc_vocab = load_kc_vocab(self.dataset_name, data_config)  # {kc_id: kc_name}
        if len(self.kc_vocab) == 0:
            print(f"[ThinkKT] 警告: 知识点词表为空，CoT生成时将使用默认的知识点ID")
        
        # 初始化知识状态追踪器
        kt_config = {
            'd_question': self.d_question,
            'd_cot': self.d_cot,
            'num_c': self.num_c,
            'd_knowledge': config.get('d_knowledge', 512),
            'dropout': config.get('dropout', 0.1),
            'use_cot': self.use_cot,
            'seq_model_type': config.get('seq_model_type', 'transformer'),
            'num_transformer_layers': config.get('num_transformer_layers', 6),
            'num_heads': config.get('num_heads', 8),
            'num_lstm_layers': config.get('num_lstm_layers', 2)
        }
        
        print(f"[ThinkKT] 正在初始化知识状态追踪器...")
        self.kt_net = ThinkKTNet(kt_config, data_config)
        
        # --- Meta-Controller (RL Policy Network) ---
        # 输入: 状态特征 (d_question + d_knowledge + stat_features)
        # 这里为了简化，我们假设状态由题目特征和当前知识状态组成
        # 实际输入维度需要根据 _get_policy_state 方法的输出决定
        # 暂时设定输入维度为 d_question + d_knowledge + 1 (频次特征)
        self.d_policy_state = self.d_question + config.get('d_knowledge', 512) + 1
        
        self.meta_policy_net = nn.Sequential(
            nn.Linear(self.d_policy_state, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出生成概率 P(Action=1)
        )
        print(f"[ThinkKT] Meta-Controller 初始化完成")
        # -------------------------------------------

        # 为了兼容 train_model.py 中的 model.model.train() 调用
        # 将 kt_net 也赋值给 model 属性
        self.model = self.kt_net
        
        print(f"[ThinkKT] 模型初始化完成")
    
    def _get_question_features(
        self, 
        qids: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        获取题目特征
        
        Args:
            qids: 问题ID张量 (batch_size, seq_len)
            seq_len: 序列长度
            
        Returns:
            v_t: 题目特征 (batch_size, seq_len, d_question)
        """
        # 1. 获取 QID Embedding (200 dim)
        v_qid = None
        if self.QEmbs is not None:
             # 处理 Padding: 将负数索引 (如 -1) 替换为 0
             safe_qids = qids.long()
             safe_qids = torch.where(safe_qids >= 0, safe_qids, torch.zeros_like(safe_qids))
             safe_qids = torch.clamp(safe_qids, 0, self.num_q - 1)
             v_qid = self.QEmbs(safe_qids) # (batch, seq, 200)

        # 2. 获取 Visual Feature (1024 -> 200 dim)
        v_visual = None
        if self.visual_encoder is not None and self.visual_projector is not None:
            # 使用多模态编码器提取原生特征 (batch, seq, 1024)
            v_raw, _ = self.visual_encoder(qids, self.img_path_dict, return_kc=False)
            # 投影降维
            v_visual = self.visual_projector(v_raw) # (batch, seq, 200)
        elif self.use_visual and self.visual_projector is not None:
             # Visual mode but encoder logic not ready? (Fallback)
             # Should assume visual_encoder is present if use_visual=True
             pass
             
        # 3. 组合特征
        if self.question_rep_type == 'qid':
            return v_qid
        elif self.question_rep_type == 'visual':
            if v_visual is None:
                # Fallback: Zero vector of target dim
                return torch.zeros((qids.shape[0], seq_len, self.net_d_question), device=self.device)
            return v_visual
        elif self.question_rep_type == 'v&q':
            if v_qid is None or v_visual is None:
                 # Should not happen given init logic
                 raise ValueError("v&q mode requires both QID and Visual features")
            return torch.cat([v_qid, v_visual], dim=-1) # (batch, seq, 400)
            
        return v_qid # Fallback
    
    def _get_cot_embeddings(
        self,
        qseqs: torch.Tensor,
        rseqs: torch.Tensor,
        cseqs: Optional[torch.Tensor] = None,
        img_path_dict=None,
        kc_vocab=None,
        v_feat=None
    ) -> Optional[torch.Tensor]:
        """
        获取CoT嵌入
        
        Args:
            qids: 问题ID张量 (batch_size, seq_len)
            rseqs: 答题结果张量 (batch_size, seq_len)
            cseqs: 知识点序列 (batch_size, seq_len, max_concepts) 可选
            img_path_dict: 问题ID到图片路径的映射
            kc_vocab: 知识点词表
            v_feat: (batch, seq_len, d_question) 题目视觉特征，若为 None 则内部暂无法使用策略网络(除非再次计算)
            
        Returns:
            r_embed: CoT嵌入 (batch_size, seq_len, d_cot) 或 None
        """
        if not self.use_cot or self.cot_generator is None:
            return None
        
        batch_size, seq_len = qseqs.shape
        device = qseqs.device
        
        cot_embeds = []
        
        # 维护知识点频次 (batch_level)
        # map: batch_idx -> {kc: count}
        from collections import defaultdict
        kc_counts_map = [defaultdict(int) for _ in range(batch_size)]
        
        # 统计 Counters
        processed_items = 0
        cached_count = 0
        generated_count = 0
        total_items = batch_size * seq_len
        # 转换为 list 以便处理
        qseqs_list = qseqs.cpu().tolist()
        rseqs_list = rseqs.cpu().tolist()
        if cseqs is not None:
            cseqs_list = cseqs.cpu().tolist()
        
        print(f"[ThinkKT] Generating CoT (Strategy={self.adaptive_strategy})...")
        import sys
        sys.stdout.flush()
        
        for i in range(batch_size):
            history_qids = []
            history_rs = []
            history_kcs_list = [] # List of lists for history
            
            # 序列当前步的 CoT 列表
            seq_cot_embeds = []
            
            for j in range(seq_len):
                qid = qseqs_list[i][j]
                if qid == 0: # Padding
                    seq_cot_embeds.append(torch.zeros(self.d_cot).to(device))
                    # Update history for next step, but don't process padding
                    if j > 0: # Only add to history if not the first element and not padding
                        history_qids.append(qseqs_list[i][j-1])
                        history_rs.append(rseqs_list[i][j-1])
                        if cseqs is not None:
                            prev_kcs = [k for k in cseqs_list[i][j-1] if k != -1]
                            history_kcs_list.append(prev_kcs)
                    continue
                    
                processed_items += 1
                
                # 获取知识点
                current_kcs = []
                if cseqs is not None:
                    # 假设 cseqs: [batch, seq_len, max_concepts]
                    # 我们过滤掉 -1 padding
                    for k in cseqs_list[i][j]:
                        if k != -1:
                            current_kcs.append(k)
                
                # --- 核心决策逻辑 ---
                should_generate = False
                
                # 1. 计算频次状态
                # 这里简单取第一个知识点作为代表，或者取最大频次
                min_count = 9999
                if current_kcs:
                    for k in current_kcs:
                        c = kc_counts_map[i][k]
                        if c < min_count: min_count = c
                else: # If no KCs, treat as novel
                    min_count = 0
                
                # 2. 策略分支
                if self.adaptive_strategy == 'rule':
                    # 规则模式: 频次小于阈值则生成
                    if min_count < self.cot_threshold:
                        should_generate = True
                        
                elif self.adaptive_strategy == 'learnable':
                    # RL 模式: 使用 Policy Network 判断
                    # 注意: 为了推理时能用，我们需要 v_t。
                    # 如果 v_feat 传进来了，就用；否则暂时降级为 Rule (防止报错)
                    if v_feat is not None:
                        # 构造 Input
                        v_t_current = v_feat[i, j].unsqueeze(0) # (1, d_question)
                        # h_t is the knowledge state *before* processing current item.
                        # For now, we use a dummy zero vector for h_prev as it's not readily available here.
                        # A more sophisticated approach would require passing h_t from KTNet.
                        h_dummy = torch.zeros(1, self.kt_net.d_knowledge).to(device) 
                        cnt_tsr = torch.tensor([[min_count]], dtype=torch.float).to(device)
                        
                        # Forward Policy
                        # 只需要概率，不需要采样 (推理时使用 Greedy 或 Threshold 0.5)
                        _, _, prob = self.forward_policy(v_t_current, h_dummy, cnt_tsr)
                        if prob.item() > 0.5:
                            should_generate = True
                    else:
                        # Fallback to rule-based if v_feat is not provided for learnable strategy
                        if min_count < self.cot_threshold: should_generate = True
                # -------------------
                
                
                # 获取当前题目图片路径
                if qid in self.img_path_dict:
                    img_path = self.img_path_dict[qid]
                else:
                    # 如果找不到路径，返回零向量
                    seq_cot_embeds.append(torch.zeros(self.d_cot, device=device))
                    if processed_items % 10 == 0:
                        print(f"[ThinkKT] CoT进度: {processed_items}/{total_items} ({100*processed_items/total_items:.1f}%) | 缓存:{cached_count} 生成:{generated_count}", end='\r')
                        sys.stdout.flush()
                    continue
                
                # 生成 CoT (仅当策略决定生成时)
                if should_generate:
                    try:
                        # 检查是否在缓存中
                        cache_key = self.cot_generator._get_cache_key(history_qids, history_rs, qid)
                        from_cache = cache_key in self.cot_generator.cot_cache
                        
                        cot_text, cot_embed = self.cot_generator.generate_cot(
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
                        
                        seq_cot_embeds.append(cot_embed.to(device))
                        
                        # 每10个或每生成一个非缓存的CoT时输出进度
                        if processed_items % 10 == 0 or not from_cache:
                            print(f"[ThinkKT] CoT进度: {processed_items}/{total_items} ({100*processed_items/total_items:.1f}%) | 缓存:{cached_count} 生成:{generated_count} | 当前qid={qid}", end='\r')
                            sys.stdout.flush()
                            
                    except Exception as e:
                        print(f"\n[ThinkKT] 警告: 生成 CoT 失败 (qid={qid}, batch={i}, seq={j}): {e}")
                        sys.stdout.flush()
                        seq_cot_embeds.append(torch.zeros(self.d_cot, device=device))
                else:
                    # 策略决定跳过 CoT 生成
                    seq_cot_embeds.append(torch.zeros(self.d_cot, device=device))
                
                # 更新知识点频次
                if current_kcs:
                    for k in current_kcs:
                        kc_counts_map[i][k] += 1
            
            cot_embeds.append(torch.stack(seq_cot_embeds))
        
        print(f"\n[ThinkKT] CoT生成完成: 总计 {processed_items} 个，缓存: {cached_count}，新生成: {generated_count}")
        sys.stdout.flush()
        
        # 堆叠为 (batch_size, seq_len, d_cot)
        r_embed = torch.stack(cot_embeds)
        return r_embed

    def _get_policy_state(self, v_t, h_prev, kc_counts_feature):
        """
        构建策略网络的输入状态
        Args:
            v_t: 题目特征 (batch, d_question)
            h_prev: 上一时刻的隐状态 (batch, d_knowledge)
            kc_counts_feature: 知识点频次特征 (batch, 1)
        Returns:
            state: (batch, d_policy_state)
        """
        return torch.cat([v_t, h_prev, kc_counts_feature], dim=-1)

    def forward_policy(self, v_t, h_prev, kc_counts):
        """
        前向传播策略网络
        Returns:
            action: (batch, 1) 采样动作 {0, 1}
            log_prob: (batch, 1) 动作的对数概率
            action_prob: (batch, 1) 生成概率 P(action=1)
        """
        # 归一化频次特征
        counts_feat = (kc_counts / 10.0).unsqueeze(-1)  # 简单归一化
        
        state = self._get_policy_state(v_t, h_prev, counts_feat)
        prob = self.meta_policy_net(state)
        
        # 采样动作
        m = torch.distributions.Bernoulli(prob)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        return action, log_prob, prob
    
    def train_one_step(self, data: dict) -> tuple:
        """
        pykt 标准接口：训练一步
        
        Args:
            data: 数据字典，包含：
                - qseqs: (batch, seq_len-1) 问题ID序列
                - cseqs: (batch, seq_len-1, max_concepts) 知识点序列
                - rseqs: (batch, seq_len-1) 答题结果序列
                - shft_qseqs: (batch, seq_len-1) 下一个问题ID序列
                - shft_rseqs: (batch, seq_len-1) 下一个答题结果序列（标签）
                - smasks: (batch, seq_len-1) 选择掩码
                - masks: (batch, seq_len-1) 掩码序列
                
        Returns:
            y: 预测结果 (batch, seq_len-1)
            loss: 损失值
        """
        # 获取输入数据
        qseqs = data['qseqs']  # (batch, seq_len-1)
        rseqs = data['rseqs']  # (batch, seq_len-1)
        shft_qseqs = data['shft_qseqs']  # (batch, seq_len-1)
        shft_rseqs = data['shft_rseqs']  # (batch, seq_len-1) 标签
        smasks = data['smasks']  # (batch, seq_len-1)
        masks = data.get('masks', None)  # (batch, seq_len-1)
        
        batch_size, seq_len = qseqs.shape
        
        # 移动到设备
        qseqs = qseqs.to(self.device)
        rseqs = rseqs.to(self.device)
        shft_qseqs = shft_qseqs.to(self.device)
        shft_rseqs = shft_rseqs.to(self.device)
        smasks = smasks.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        # 获取题目特征（使用历史问题序列）
        v_t = self._get_question_features(qseqs, seq_len)
        
        # 获取知识点序列
        cseqs = data.get('cseqs', None)  # 知识点序列 (batch, seq_len, max_concepts)
        if cseqs is None:
            # 如果没有知识点序列，创建零填充
            cseqs = torch.full((batch_size, seq_len, 1), -1, device=self.device)
        else:
            cseqs = cseqs.to(self.device)
        
        # 获取CoT嵌入
        r_embed = self._get_cot_embeddings(qseqs, rseqs, cseqs, img_path_dict=None, kc_vocab=None, v_feat=v_t)
        
        # 前向传播
        y = self.kt_net(
            v_t=v_t,
            c=cseqs,  # 知识点序列
            r=rseqs,  # 使用历史答题结果
            r_embed=r_embed,
            mask=masks,
            q_shift=shft_qseqs
        )  # (batch, seq_len-1)
        
        # 计算损失
        loss = self.get_loss(y, shft_rseqs, smasks)
        
        return y, loss

    def forward_rl(self, data: dict):
        """
        适用于强化学习训练的前向传播
        支持逐步决策和梯度流
        Returns:
            predictions: (batch, seq_len)
            actions: (batch, seq_len)
            log_probs: (batch, seq_len)
        """
        # 解包数据
        qseqs = data['qseqs']  # (batch, seq_len)
        rseqs = data['rseqs']
        cseqs = data['cseqs']
        batch_size, seq_len = qseqs.shape
        device = qseqs.device
        
        # 1. 获取题目特征 (batch, seq_len, d_question)
        v, k = self._get_question_features(qseqs, seq_len)
        
        # 2. 初始化 KT 隐状态
        h_t = self.kt_net.init_hidden(batch_size).to(device) # 需要确保 ThinkKTNet 有这个方法
        
        predictions_list = []
        actions_list = []
        log_probs_list = []
        
        # 维护知识点频次
        kc_counts_map = [defaultdict(int) for _ in range(batch_size)]
        
        # 3. 逐步循环
        for t in range(seq_len):
            # 当前时刻输入
            v_t = v[:, t, :] # (batch, d_question)
            r_prev = torch.zeros(batch_size, dtype=torch.long).to(device) if t == 0 else rseqs[:, t-1]
            q_t_id = qseqs[:, t]
            
            # --- 准备策略状态 ---
            # 计算当前题目主要知识点的频次
            current_counts = []
            for b in range(batch_size):
                # 获取当前题目的主要知识点（取第一个作为代表，或取平均）
                # 这里简化：假设cseqs[:,:,0]是主要知识点
                kc = int(cseqs[b, t, 0].item()) if cseqs is not None else -1
                cnt = kc_counts_map[b][kc]
                current_counts.append(cnt)
                kc_counts_map[b][kc] += 1
            
            kc_counts_tensor = torch.tensor(current_counts, dtype=torch.float, device=device).unsqueeze(1) # (batch, 1)
            
            # --- Meta-Controller 决策 ---
            # 注意：这里需要 h_t，假设 h_t 是 (batch, d_knowledge)
            # 如果是 LSTM 的 h_t 是 tuple，需要处理
            if isinstance(h_t, tuple):
                 h_prev_Rep = h_t[0][-1] # 取最后一层的隐藏状态
            else:
                 h_prev_Rep = h_t
                 
            action_t, log_prob_t, _ = self.forward_policy(v_t, h_prev_Rep, kc_counts_tensor)
            
            actions_list.append(action_t)
            log_probs_list.append(log_prob_t)
            
            # --- 执行 CoT (Action=1) 或 Skip (Action=0) ---
            # 注意：实际生成文本很慢，RL训练时我们通常预计算好或者使用缓存
            # 这里为了演示完整流，我们根据 action 选择
            
            # 由于这是必须要梯度的过程，我们不能直接调用 _get_cot_embeddings (它不传梯度)
            # 简化方案：RL阶段只训练 Policy，冻结 Generator 和 KT
            # 因此我们在此处只需要获取 CoT embedding
            
            # *关键*: 我们需要能根据 action 动态获得 embedding
            # 暂时用一个 placeholder 函数，实际需要根据 dataset 获取
            # 为了跑通流程，这里先调用 _get_cot_embeddings 的改进版（需修改），或者在此处手动构建
            
            # 临时：构造一个伪 CoT 向量 (如果 action=1)
            # 真实场景：调用 self.cot_generator.generate_cot_embedding_only(...)
            cot_embed_t = torch.zeros(batch_size, self.d_model, device=device) # 假设 d_model = d_cot
            
            # 如果 action=1，且有预计算的 CoT，则填入
            # 这里留空给后续对接
            
            # --- KT 更新 ---
            # 调用 KT Net 的单步 forward
            # y_t, h_next = self.kt_net.forward_step(v_t, r_prev, cot_embed_t, h_t) 
            # 假设 ThinkKTNet 还没有 forward_step，需要增加
            
            # 为了不报错，先占位
            predictions_list.append(torch.zeros(batch_size, 1, device=device)) 
            
        return torch.stack(predictions_list, dim=1), torch.stack(actions_list, dim=1), torch.stack(log_probs_list, dim=1)
    
    def predict_one_step(self, data: dict) -> torch.Tensor:
        """
        pykt 标准接口：预测一步
        
        Args:
            data: 数据字典（与train_one_step相同）
            
        Returns:
            y: 预测结果 (batch, seq_len-1)
        """
        with torch.no_grad():
            y, _ = self.train_one_step(data)
        return y
    
    def get_loss(
        self, 
        ys: torch.Tensor, 
        rshft: torch.Tensor, 
        sm: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            ys: 预测结果 (batch, seq_len-1)
            rshft: 真实标签 (batch, seq_len-1)
            sm: 选择掩码 (batch, seq_len-1)
            
        Returns:
            loss: 损失值
        """
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        
        loss = F.binary_cross_entropy(y_pred.double(), y_true.double())
        return loss
    
    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device
        
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(device)
        
        if self.cot_generator is not None:
            self.cot_generator = self.cot_generator.to(device)
        
        self.kt_net = self.kt_net.to(device)
        
        if self.QEmbs is not None:
            self.QEmbs = self.QEmbs.to(device)
            
        return self
    
    def save_feature_cache(self):
        """保存特征缓存"""
        if self.visual_encoder is not None:
            self.visual_encoder.save_feature_cache()
        
        # 保存CoT缓存
        if self.cot_generator is not None:
            self.cot_generator._save_cot_cache()
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        重写state_dict，排除vision_model（预训练模型不需要保存）
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # 移除 vision_model 相关的参数（保留其他 visual_encoder 的参数，如 feature_proj, kc_classifier）
        keys_to_remove = []
        for key in state_dict.keys():
            if 'vision_model' in key and 'visual_encoder' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del state_dict[key]
        
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        重写load_state_dict，允许忽略vision_model相关的参数
        """
        # 过滤掉vision_model相关的参数
        filtered_state_dict = {}
        vision_model_keys = []
        
        for key, value in state_dict.items():
            if 'vision_model' in key and 'visual_encoder' in key:
                vision_model_keys.append(key)
            else:
                filtered_state_dict[key] = value
        
        if vision_model_keys:
            print(f"[ThinkKT] 跳过加载 {len(vision_model_keys)} 个 vision_model 参数（预训练模型将在初始化时加载）")
        
        # 使用strict=False加载，因为可能还有其他不匹配的参数
        return super().load_state_dict(filtered_state_dict, strict=False)

