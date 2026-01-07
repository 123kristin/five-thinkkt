import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import atexit
from typing import Dict, Optional, List

# 添加 JiT 代码路径
# Try to locate JiT relative to this file or using hardcoded paths
current_file_path = os.path.abspath(__file__)
# Go up 6 levels from .../pykt/models/our_model/jikt.py to five-thinkkt root
# pykt/models/our_model/jikt.py -> our_model -> models -> pykt -> examples -> scripts_training... -> five-thinkkt
five_thinkkt_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))))
crkt_root = os.path.dirname(five_thinkkt_root)

possible_jit_paths = [
    os.path.join(five_thinkkt_root, "JiT"), # First priority: inside five-thinkkt (user's new structure)
    os.path.join(crkt_root, "JiT"),         # Sibling directory
    os.path.join(five_thinkkt_root, "../JiT"),
    "/home3/zhiyu/code-5/CRKT/five-thinkkt/JiT", # Hardcoded check
    "/home3/zhiyu/code-5/CRKT/JiT",
    "/home/zhiyu/other_code/JiT"
]

jit_path_found = None
for p in possible_jit_paths:
    if os.path.exists(os.path.join(p, "model_jit.py")):
        jit_path_found = p
        break

if jit_path_found:
    if jit_path_found not in sys.path:
        sys.path.append(jit_path_found)
    print(f"[JiKT] Successfully added JiT path: {jit_path_found}")
else:
    print(f"[JiKT] Warning: valid JiT path not found. Checked: {possible_jit_paths}")
    # Add default just in case to avoid immediate crash if it's magically in path differently
    sys.path.append("/home3/zhiyu/code-5/CRKT/JiT")
from model_jit import JiT_B_32
from torchvision import transforms
from PIL import Image

# Import helper from visual_language_encoder
try:
    from .visual_language_encoder import build_img_path_dict
except ImportError:
    # Fallback if relative import fails when running directly
    from .visual_language_encoder import build_img_path_dict

class JiTAdapter(nn.Module):
    def __init__(self, device, dataset_name=None, cache_dir="features", use_cache=True):
        super().__init__()
        self.device = device
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 初始化 JiT-B/32 模型 (Patch Size=32, Resolutions=512)
        # 注意: 这里我们只初始化模型结构，实际应该加载预训练权重
        # 为了避免显存爆炸，我们冻结所有参数
        # JiT_B_32 already sets patch_size=32. Arg is input_size not img_size.
        self.jit_model = JiT_B_32(input_size=512)
        
        # 冻结参数
        for param in self.jit_model.parameters():
            param.requires_grad = False
        
        self.jit_model.eval() # 始终保持评估模式
        self.jit_model.to(device)

        # 预处理: 归一化 (ImageNet mean/std)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Resize 对于非 512x512 的图片 (如 NIPS 的 640x480)
        self.resize = transforms.Resize((512, 512))
        
        # 缓存系统
        self.feature_cache = {}
        if self.use_cache:
            self._load_feature_cache()
            atexit.register(self.save_feature_cache)

    def _get_cache_path(self) -> str:
        """获取特征缓存文件路径 (Distinct from Qwen cache)"""
        if self.dataset_name:
            cache_file = f"{self.dataset_name}_jit_features.pt"
        else:
            cache_file = "jit_features.pt"
        return os.path.join(self.cache_dir, cache_file)

    def _load_feature_cache(self):
        """加载特征缓存"""
        if not self.use_cache: return
        
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print(f"[JiTAdapter] 正在加载特征缓存: {cache_path}")
                self.feature_cache = torch.load(cache_path, map_location='cpu')
                print(f"[JiTAdapter] 已加载 {len(self.feature_cache)} 个题目特征")
            except Exception as e:
                print(f"[JiTAdapter] 警告: 加载缓存失败 {e}，将重新计算")
                self.feature_cache = {}

    def save_feature_cache(self):
        """保存特征缓存"""
        if not self.use_cache or not self.feature_cache: return
        
        cache_path = self._get_cache_path()
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        try:
            torch.save(self.feature_cache, cache_path)
            print(f"[JiTAdapter] 已保存 {len(self.feature_cache)} 个题目特征到 {cache_path}")
        except Exception as e:
            print(f"[JiTAdapter] 警告: 保存缓存失败 {e}")

    def preprocess(self, x):
        """
        x: [B, 3, H, W] tensor
        """
        if x.shape[-2:] != (512, 512):
            x = self.resize(x)
        x = self.normalize(x)
        return x

    def extract_feature_from_image(self, img_path):
        """读取并处理单张图片"""
        try:
            image = Image.open(img_path).convert('RGB')
            # Transform to tensor
            x = transforms.ToTensor()(image).unsqueeze(0).to(self.device) # [1, 3, H, W]
            
            # Preprocess
            x = self.preprocess(x)
            
            # JiT Forward (Frozen)
            with torch.no_grad():
                x_emb = self.jit_model.x_embedder(x)
                x_emb += self.jit_model.pos_embed
                
                t = torch.zeros(x.shape[0], device=x.device)
                y = torch.zeros(x.shape[0], device=x.device).long()
                
                t_emb = self.jit_model.t_embedder(t)
                y_emb = self.jit_model.y_embedder(y)
                c = t_emb + y_emb
                
                for block in self.jit_model.blocks:
                    x_emb = block(x_emb, c, self.jit_model.feat_rope)
                    
                # Global Average Pooling
                feature = x_emb.mean(dim=1).squeeze(0) # [C]
                
            return feature # On device
            
        except Exception as e:
            print(f"[JiTAdapter] Error processing {img_path}: {e}")
            return torch.zeros(768, device=self.device)

    def forward(self, qids, img_path_dict):
        """
        Input: 
            qids: [B, Seq] or [B*Seq] Tensor
            img_path_dict: {qid: path}
        Output:
            features: [B, Seq, 768] or [B*Seq, 768]
        """
        original_shape = qids.shape
        qids_flat = qids.view(-1).cpu().numpy()
        
        batch_features = []
        
        # 1. 查找缓存
        # 优化: 预先收集所有 miss 的 qid
        process_indices = []
        process_paths = []
        
        for i, qid in enumerate(qids_flat):
            if qid in self.feature_cache:
                batch_features.append(self.feature_cache[qid])
            else:
                batch_features.append(None) # 占位
                if qid in img_path_dict:
                    process_indices.append(i)
                    process_paths.append(img_path_dict[qid])
        
        # 2. 处理 Missing (逐个处理，为了简单和显存安全)
        # 如果需要加速，可以 mini-batch 处理
        for idx, path in zip(process_indices, process_paths):
            feat = self.extract_feature_from_image(path)
            batch_features[idx] = feat.cpu() # 存入缓存前转CPU
            
            # 更新缓存
            qid = qids_flat[idx]
            if self.use_cache:
                self.feature_cache[qid] = feat.cpu()
                
        # 3. 填充剩余 None (qid 不在 dict 中的情况，如 padding 0)
        # 默认 0 向量
        zero_vec = torch.zeros(768)
        for i in range(len(batch_features)):
            if batch_features[i] is None:
                batch_features[i] = zero_vec
        
        # 4. Stack and Move to Device
        feature_tensor = torch.stack(batch_features).to(self.device)
        
        return feature_tensor.view(*original_shape, -1)


class JiKTNet(nn.Module):
    def __init__(self, config: dict, dataset_name=None, img_path_dict=None):
        super(JiKTNet, self).__init__()
        self.model_name = 'jikt_net'
        self.dataset_name = dataset_name
        self.img_path_dict = img_path_dict or {}
        
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[JiKTNet] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[JiKTNet] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[JiKTNet] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[JiKTNet] CUDA不可用，使用CPU")
        
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

        # 能力模块
        self.dim_knowledge = self.dim_qc
        self.rnn_type = config.get('rnn_type', 'lstm')
        if self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(self.dim_qc * 4, self.dim_knowledge, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(self.dim_qc * 4, self.dim_knowledge, batch_first=True)

        self.q_scores_extractor = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_knowledge, self.num_q)
        )
        
        # --- 多模态融合模块 ---
        # 1. JiT 适配器 (带缓存)
        print(f"[JiKTNet] 初始化 JiTAdapter (数据集: {self.dataset_name})...")
        self.jit_encoder = JiTAdapter(self.device, dataset_name=self.dataset_name)
        
        # 2. 视觉投影层 (768 -> dim_qc)
        jit_hidden_dim = 768 
        self.visual_projector = nn.Sequential(
            nn.Linear(jit_hidden_dim, self.dim_qc),
            nn.ReLU(),
            nn.Linear(self.dim_qc, self.dim_qc)
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
        
        # JiTAdapter 也需要 to
        self.jit_encoder.device = device # Update internal device tracking
        self.jit_encoder.jit_model.to(device)
        
        return self

    def get_kc_avg_emb(self, c, pad_idx=-1):
        if c.dim() == 2:
            # Handle single-concept input: [bz, len]
            mask = c != pad_idx
            c_safe = c.masked_fill(~mask, 0)
            embs = self.KCEmbs(c_safe) # [bz, len, emb_size]
            embs = embs * mask.unsqueeze(-1)
            return embs

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

    def forward(self, q, c, r, q_shift, return_all=False):
        """
        Modified forward to handle image loading internally via jit_encoder
        """
        bz, num_interactions = q.shape  # num_interactions = interactions_seq_len - 1

        # 确保c_ids在正确设备上
        self._ensure_c_ids_on_device()

        # 移入 device
        q = q.to(self.device)
        c = c.to(self.device)
        r = r.to(self.device)
        q_shift = q_shift.to(self.device)

        q_emb = self.QEmbs(q)  # [bz, num_interactions, dim_qc]
        
        # --- 视觉特征融合 ---
        # 使用 QID 序列查找图片并提取特征
        # q: [bz, num_interactions]
        # features: [bz, num_interactions, 768]
        vis_feats = self.jit_encoder(q, self.img_path_dict)
            
        # 投影及恢复形状 [Bz, Seq, dim_qc]
        vis_emb = self.visual_projector(vis_feats)
            
        # 加法融合: 增强 ID Embedding
        q_emb = q_emb + vis_emb

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


class JiKT(nn.Module):
    """
    消融实验用，没有多头注意力的block 块，即没有建模q_kcs的权重以及
    """

    def __init__(self, config, data_config=None):
        super(JiKT, self).__init__()
        self.model_name = 'jikt'
        self.emb_type = config.get('emb_type', 'qkcs')
        
        # 改进设备选择逻辑，真正选择指定的GPU卡号
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    print(f"[JiKT] 使用指定GPU: cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda:0")
                    print(f"[JiKT] 指定GPU {gpu_id} 不可用，使用默认GPU: cuda:0")
            except ValueError:
                self.device = torch.device("cuda:0")
                print(f"[JiKT] GPU ID解析失败，使用默认GPU: cuda:0")
        else:
            self.device = torch.device("cpu")
            print(f"[JiKT] CUDA不可用，使用CPU")
        
        # 构建图片路径映射
        self.dataset_name = config.get('dataset_name', 'DBE_KT22')
        # 如果 data_config 有 dpath，从文件名推断 dataset_name (ThinkKT logic)
        if data_config:
             dpath = data_config.get('dpath', '')
             if 'XES3G5M' in dpath: self.dataset_name = 'XES3G5M'
             elif 'DBE_KT22' in dpath: self.dataset_name = 'DBE_KT22'
             elif 'NIPS_task34' in dpath: self.dataset_name = 'NIPS_task34'
        
        img_path_dict = {}
        if data_config:
            print(f"[JiKT] Building image path dict for {self.dataset_name}...")
            img_path_dict = build_img_path_dict(self.dataset_name, data_config)
            print(f"[JiKT] Loaded {len(img_path_dict)} images.")
        else:
            print(f"[JiKT] Warning: No data_config provided, image features will be zero.")

        self.model = JiKTNet(config, dataset_name=self.dataset_name, img_path_dict=img_path_dict)

    def train_one_step(self, data):
        # 无需手动传 img_seq，内部根据 qseqs 自动查找缓存
        y = self.model(data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'])

        sm = data['smasks'].to(self.device)
        r_shift = data['shft_rseqs'].to(self.device)
        # calculate loss
        loss = self.get_loss(y, r_shift, sm)

        return y, loss

    def predict_one_step(self, data):
        y = self.model(data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'])
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
    model = JiKT(config)
    print(model)
