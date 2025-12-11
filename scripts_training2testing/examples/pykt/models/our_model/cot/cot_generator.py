"""
知识推理链生成器（Knowledge CoT Generator）
使用 MLLM 生成思维链推理
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import json
import hashlib
from functools import lru_cache
import sys

# 导入 transformers
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None
    Image = None

# 导入文本编码器（用于编码 CoT 文本）
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .cot_prompts import build_cot_prompt, parse_cot_response, validate_cot


class CoTGenerator(nn.Module):
    """
    知识推理链生成器
    
    功能：
    1. 使用 MLLM 生成 CoT 文本
    2. 使用文本编码器编码 CoT
    3. CoT 缓存管理
    """
    
    def __init__(
        self,
        mllm_name: str = "/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
        text_encoder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        d_cot: int = 384,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = True
    ):
        """
        初始化 CoT 生成器
        
        Args:
            mllm_name: MLLM 模型路径
            text_encoder_name: 文本编码器名称
            d_cot: CoT 嵌入维度
            cache_dir: 缓存目录
            device: 设备
            use_cache: 是否使用缓存
        """
        super(CoTGenerator, self).__init__()
        
        # 设备管理
        if device is None:
            device = self._get_device()
        self.device = device
        print(f"[CoTGenerator] 使用设备: {self.device}")
        
        self.mllm_name = mllm_name
        self.text_encoder_name = text_encoder_name
        self.d_cot = d_cot
        self.use_cache = use_cache
        
        # 初始化 MLLM（延迟加载）
        self.mllm_model = None
        self.mllm_processor = None
        self._mllm_loaded = False
        
        # 初始化文本编码器（延迟加载）
        self.text_encoder = None
        self._text_encoder_loaded = False
        
        # CoT 缓存
        self.cache_dir = cache_dir if cache_dir else "cot_cache"
        self.cot_cache = {}
        self._load_cot_cache()
    
    def _get_device(self):
        """获取设备"""
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    return torch.device(f"cuda:{gpu_id}")
                else:
                    return torch.device("cuda:0")
            except ValueError:
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    def _load_mllm(self):
        """延迟加载 MLLM"""
        if self._mllm_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("无法导入 transformers 库")
        
        print(f"[CoTGenerator] 正在加载 MLLM: {self.mllm_name}")
        try:
            self.mllm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.mllm_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.mllm_processor = AutoProcessor.from_pretrained(
                self.mllm_name,
                trust_remote_code=True
            )
            self.mllm_model.eval()
            self._mllm_loaded = True
            print(f"[CoTGenerator] MLLM 加载完成")
        except Exception as e:
            raise RuntimeError(f"加载 MLLM 失败: {e}")
    
    def _load_text_encoder(self):
        """延迟加载文本编码器"""
        if self._text_encoder_loaded:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[CoTGenerator] 警告: sentence-transformers 未安装，将使用简单的文本编码")
            self.text_encoder = None
            self._text_encoder_loaded = True
            return
        
        print(f"[CoTGenerator] 正在加载文本编码器: {self.text_encoder_name}")
        try:
            self.text_encoder = SentenceTransformer(self.text_encoder_name, device=str(self.device))
            self._text_encoder_loaded = True
            print(f"[CoTGenerator] 文本编码器加载完成")
        except Exception as e:
            print(f"[CoTGenerator] 警告: 加载文本编码器失败: {e}，将使用简单编码")
            self.text_encoder = None
            self._text_encoder_loaded = True
    
    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, "cot_cache.jsonl")
    
    def _load_cot_cache(self):
        """加载 CoT 缓存"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print(f"[CoTGenerator] 正在加载 CoT 缓存: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            cache_key = item.get('cache_key')
                            if cache_key:
                                self.cot_cache[cache_key] = item
                print(f"[CoTGenerator] 已加载 {len(self.cot_cache)} 个 CoT 缓存")
            except Exception as e:
                print(f"[CoTGenerator] 警告: 加载缓存失败 {e}")
                self.cot_cache = {}
        else:
            self.cot_cache = {}
    
    def _save_cot_cache(self):
        """保存 CoT 缓存"""
        if not self.use_cache or not self.cot_cache:
            return
        
        cache_path = self._get_cache_path()
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                for item in self.cot_cache.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"[CoTGenerator] 已保存 {len(self.cot_cache)} 个 CoT 到缓存")
        except Exception as e:
            print(f"[CoTGenerator] 警告: 保存缓存失败 {e}")
    
    def _get_cache_key(
        self,
        history_qids: List[int],
        history_rs: List[int],
        current_qid: int
    ) -> str:
        """生成缓存键"""
        history_str = f"{','.join(map(str, history_qids))}:{','.join(map(str, history_rs))}"
        key_str = f"{history_str}|{current_qid}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def generate_cot(
        self,
        history_qids: List[int],
        history_rs: List[int],
        current_qid: int,
        img_path: str,
        kc_vocab: Dict[int, str],
        history_kcs: Optional[List[List[int]]] = None,
        current_kcs: Optional[List[int]] = None
    ) -> Tuple[str, torch.Tensor]:
        """
        生成 CoT 文本和嵌入
        
        Args:
            history_qids: 历史问题ID列表
            history_rs: 历史答题结果列表
            current_qid: 当前问题ID
            img_path: 当前题目图片路径
            kc_vocab: 知识点词表
            history_kcs: 历史问题的知识点列表（可选）
            current_kcs: 当前问题的知识点列表（可选）
            
        Returns:
            cot_text: CoT 文本
            cot_embed: CoT 嵌入向量 (d_cot,)
        """
        # 检查缓存
        cache_key = self._get_cache_key(history_qids, history_rs, current_qid)
        if cache_key in self.cot_cache:
            cached_item = self.cot_cache[cache_key]
            cot_text = cached_item.get('cot_text', '')
            cot_embed = torch.tensor(cached_item.get('cot_embed', []), device=self.device)
            if cot_text and cot_embed.numel() > 0:
                return cot_text, cot_embed
        
        # 加载模型
        if not self._mllm_loaded:
            self._load_mllm()
        
        # 构建 prompt
        prompt = build_cot_prompt(
            history_qids, history_rs, current_qid,
            kc_vocab, history_kcs, current_kcs
        )
        
        # 生成 CoT
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            
            # 准备输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理输入
            text = self.mllm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = [image]
                video_inputs = []
            
            # 构建 processor 参数（如果 video_inputs 为空，则不传入 videos 参数）
            processor_kwargs = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt"
            }
            # 只有当 video_inputs 不为空时才添加 videos 参数
            if video_inputs and len(video_inputs) > 0:
                processor_kwargs["videos"] = video_inputs
            
            inputs = self.mllm_processor(**processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                generated_ids = self.mllm_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                cot_text = self.mllm_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            
            # 验证和清理 CoT
            cot_text = cot_text.strip()
            if not validate_cot(cot_text):
                print(f"[CoTGenerator] 警告: 生成的 CoT 不符合要求，使用默认文本")
                cot_text = f"学生历史答题记录显示{'掌握' if sum(history_rs) > len(history_rs)/2 else '薄弱'}相关知识点。当前题目考察知识点：{', '.join([kc_vocab.get(kc, '未知') for kc in (current_kcs or [])])}。"
            
            # 编码 CoT
            cot_embed = self.encode_cot(cot_text)
            
            # 缓存
            if self.use_cache:
                self.cot_cache[cache_key] = {
                    'cache_key': cache_key,
                    'cot_text': cot_text,
                    'cot_embed': cot_embed.cpu().tolist(),
                    'history_qids': history_qids,
                    'history_rs': history_rs,
                    'current_qid': current_qid
                }
            
            return cot_text, cot_embed
            
        except Exception as e:
            print(f"[CoTGenerator] 警告: 生成 CoT 失败: {e}")
            # 返回默认 CoT
            default_cot = f"学生历史答题记录显示{'掌握' if sum(history_rs) > len(history_rs)/2 else '薄弱'}相关知识点。"
            cot_embed = self.encode_cot(default_cot)
            return default_cot, cot_embed
    
    def encode_cot(self, cot_text: str) -> torch.Tensor:
        """
        编码 CoT 文本为向量
        
        Args:
            cot_text: CoT 文本
            
        Returns:
            cot_embed: CoT 嵌入向量 (d_cot,)
        """
        if not self._text_encoder_loaded:
            self._load_text_encoder()
        
        if self.text_encoder is not None:
            # 使用 sentence-transformers
            with torch.no_grad():
                embed = self.text_encoder.encode(cot_text, convert_to_tensor=True)
                # 投影到目标维度
                if embed.shape[0] != self.d_cot:
                    # 简单的线性投影（如果需要）
                    if not hasattr(self, '_embed_proj'):
                        import torch.nn as nn
                        self._embed_proj = nn.Linear(embed.shape[0], self.d_cot).to(self.device)
                    embed = self._embed_proj(embed)
                return embed.squeeze(0) if embed.dim() > 1 else embed
        else:
            # 简单的字符级编码（fallback）
            import hashlib
            hash_obj = hashlib.md5(cot_text.encode())
            hash_bytes = hash_obj.digest()
            # 扩展到 d_cot 维度
            embed = torch.tensor(list(hash_bytes) * (self.d_cot // len(hash_bytes) + 1)[:self.d_cot], 
                                dtype=torch.float32, device=self.device)
            return embed / embed.norm() * 10  # 归一化
    
    def to(self, device):
        """重写to方法"""
        super().to(device)
        self.device = device
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device)
        return self

