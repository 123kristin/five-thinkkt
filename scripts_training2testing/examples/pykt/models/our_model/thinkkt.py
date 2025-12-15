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

# 导入自定义模块
from .visual_language_encoder import VisualLanguageEncoder, build_img_path_dict
from .thinkkt_net import ThinkKTNet
try:
    from .cot.cot_generator import CoTGenerator
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    CoTGenerator = None


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
                    use_cache=True
                )
                print(f"[ThinkKT] CoT 生成器初始化完成")
                sys.stdout.flush()
        else:
            self.cot_generator = None
        
        # 构建知识点词表（需要从数据配置中获取）
        # TODO: 从实际数据中加载知识点词表
        self.kc_vocab = {}  # {kc_id: kc_name}
        
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
        self.kt_net = ThinkKTNet(kt_config)
        
        # 为了兼容 train_model.py 中的 model.model.train() 调用
        # 将 kt_net 也赋值给 model 属性
        self.model = self.kt_net
        
        print(f"[ThinkKT] 模型初始化完成")
    
    def _get_question_features(
        self, 
        qids: torch.Tensor,
        seq_len: int
    ) -> tuple:
        """
        获取题目特征和知识点分布
        
        Args:
            qids: 问题ID张量 (batch_size, seq_len)
            seq_len: 序列长度
            
        Returns:
            v_t: 题目特征 (batch_size, seq_len, d_question)
            k_t: 知识点分布 (batch_size, seq_len, num_c)
        """
        if not self.use_visual or self.visual_encoder is None:
            # 如果不使用视觉特征，返回零向量
            batch_size = qids.shape[0]
            v_t = torch.zeros(
                (batch_size, seq_len, self.d_question),
                device=self.device
            )
            k_t = torch.zeros(
                (batch_size, seq_len, self.num_c),
                device=self.device
            )
            return v_t, k_t
        
        # 使用多模态编码器提取特征
        v_t, k_t = self.visual_encoder(
            qids,
            self.img_path_dict,
            return_kc=True
        )
        
        return v_t, k_t
    
    def _get_cot_embeddings(
        self,
        qids: torch.Tensor,
        rseqs: torch.Tensor,
        cseqs: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        获取CoT嵌入
        
        Args:
            qids: 问题ID张量 (batch_size, seq_len)
            rseqs: 答题结果张量 (batch_size, seq_len)
            cseqs: 知识点序列 (batch_size, seq_len, max_concepts) 可选
            
        Returns:
            r_embed: CoT嵌入 (batch_size, seq_len, d_cot) 或 None
        """
        if not self.use_cot or self.cot_generator is None:
            return None
        
        batch_size, seq_len = qids.shape
        device = qids.device
        
        # 将qids移到CPU进行路径查找
        qids_cpu = qids.cpu().numpy()
        rseqs_cpu = rseqs.cpu().numpy()
        
        # 批量生成 CoT 嵌入
        cot_embeds = []
        total_items = batch_size * seq_len
        processed_items = 0
        cached_count = 0
        generated_count = 0
        
        print(f"[ThinkKT] 开始生成CoT嵌入: batch_size={batch_size}, seq_len={seq_len}, 总计 {total_items} 个")
        import sys
        sys.stdout.flush()
        
        for b in range(batch_size):
            batch_cot_embeds = []
            for s in range(seq_len):
                qid = int(qids_cpu[b, s])
                processed_items += 1
                
                # 获取历史交互
                history_qids = [int(qids_cpu[b, i]) for i in range(s)]
                history_rs = [int(rseqs_cpu[b, i]) for i in range(s)]
                
                # 获取当前题目图片路径
                if qid in self.img_path_dict:
                    img_path = self.img_path_dict[qid]
                else:
                    # 如果找不到路径，返回零向量
                    batch_cot_embeds.append(torch.zeros(self.d_cot, device=device))
                    if processed_items % 10 == 0:
                        print(f"[ThinkKT] CoT进度: {processed_items}/{total_items} ({100*processed_items/total_items:.1f}%) | 缓存:{cached_count} 生成:{generated_count}", end='\r')
                        sys.stdout.flush()
                    continue
                
                # 获取知识点信息
                history_kcs = None
                current_kcs = None
                if cseqs is not None:
                    cseqs_cpu = cseqs.cpu().numpy()
                    history_kcs = [[int(cseqs_cpu[b, i, j]) for j in range(cseqs.shape[2]) 
                                   if cseqs_cpu[b, i, j] >= 0] for i in range(s)]
                    current_kcs = [int(cseqs_cpu[b, s, j]) for j in range(cseqs.shape[2]) 
                                  if cseqs_cpu[b, s, j] >= 0]
                
                # 生成 CoT
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
                        history_kcs=history_kcs,
                        current_kcs=current_kcs
                    )
                    
                    if from_cache:
                        cached_count += 1
                    else:
                        generated_count += 1
                    
                    batch_cot_embeds.append(cot_embed.to(device))
                    
                    # 每10个或每生成一个非缓存的CoT时输出进度
                    if processed_items % 10 == 0 or not from_cache:
                        print(f"[ThinkKT] CoT进度: {processed_items}/{total_items} ({100*processed_items/total_items:.1f}%) | 缓存:{cached_count} 生成:{generated_count} | 当前qid={qid}", end='\r')
                        sys.stdout.flush()
                        
                except Exception as e:
                    print(f"\n[ThinkKT] 警告: 生成 CoT 失败 (qid={qid}, batch={b}, seq={s}): {e}")
                    sys.stdout.flush()
                    batch_cot_embeds.append(torch.zeros(self.d_cot, device=device))
            
            cot_embeds.append(torch.stack(batch_cot_embeds))
        
        print(f"\n[ThinkKT] CoT生成完成: 总计 {processed_items} 个，缓存: {cached_count}，新生成: {generated_count}")
        sys.stdout.flush()
        
        # 堆叠为 (batch_size, seq_len, d_cot)
        r_embed = torch.stack(cot_embeds)
        return r_embed
    
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
        v_t, k_t = self._get_question_features(qseqs, seq_len)
        
        # 获取CoT嵌入
        cseqs = data.get('cseqs', None)  # 知识点序列（可选）
        r_embed = self._get_cot_embeddings(qseqs, rseqs, cseqs)
        
        # 前向传播
        y = self.kt_net(
            v_t=v_t,
            a_t=rseqs,  # 使用历史答题结果
            k_t=k_t,
            r_embed=r_embed,
            mask=masks
        )  # (batch, seq_len-1)
        
        # 计算损失
        loss = self.get_loss(y, shft_rseqs, smasks)
        
        return y, loss
    
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

