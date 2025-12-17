"""
多模态题目编码器（Visual-Language Encoder）
用于提取题目图片的视觉特征和知识点分布
"""
import os
import sys
import warnings
import logging

# 抑制transformers的警告信息
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('transformers').setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import pickle
import json
from functools import lru_cache

# 直接导入 transformers 库来使用 Qwen2.5-VL 模型
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
    # 尝试导入 qwen_vl_utils（可选，用于处理视觉信息）
    try:
        from qwen_vl_utils import process_vision_info
        QWEN_VL_UTILS_AVAILABLE = True
    except ImportError:
        QWEN_VL_UTILS_AVAILABLE = False
        process_vision_info = None
except ImportError:
    print("警告: transformers 库未安装，请运行: pip install transformers")
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None
    Image = None
    TRANSFORMERS_AVAILABLE = False
    QWEN_VL_UTILS_AVAILABLE = False
    process_vision_info = None


class VisualLanguageEncoder(nn.Module):
    """
    多模态题目编码器
    
    功能：
    1. 使用 Qwen2.5-VL 提取图像特征
    2. 预测知识点分布
    3. 特征缓存管理
    """
    
    def __init__(
        self,
        num_c: int = 100,
        d_question: int = 1024,
        model_path: str = "/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        use_cache: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        初始化多模态编码器
        
        Args:
            num_c: 知识点数量
            d_question: 题目特征维度
            model_path: 视觉模型路径
            cache_dir: 特征缓存目录
            dataset_name: 数据集名称（用于缓存文件命名）
            use_cache: 是否使用特征缓存
            device: 设备
        """
        super(VisualLanguageEncoder, self).__init__()
        
        # 设备管理（与CRKT一致）
        if device is None:
            device = self._get_device()
        self.device = device
        print(f"[VisualLanguageEncoder] 使用设备: {self.device}")
        
        self.num_c = num_c
        self.d_question = d_question
        self.dataset_name = dataset_name
        self.use_cache = use_cache
        
        # 初始化视觉模型（延迟加载）
        self.vision_model = None
        self.vision_processor_tokenizer = None
        self.model_path = model_path
        self._vision_model_loaded = False
        
        # 知识点分类头
        self.kc_classifier = nn.Sequential(
            nn.Linear(d_question, d_question // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_question // 2, num_c),
            nn.Sigmoid()  # 输出知识点分布（0-1之间）
        )
        
        # 特征投影层（将视觉特征投影到目标维度）
        # Qwen2.5-VL-3B 的 hidden_size 是 2048（从 config.json 中确认）
        self.feature_proj = nn.Linear(2048, d_question)
        
        # 特征缓存
        self.cache_dir = cache_dir if cache_dir else "features"
        self.question_features_cache = {}
        self._load_feature_cache()
    
    def _get_device(self):
        """获取设备（与CRKT一致）"""
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
    
    def _load_vision_processor(self):
        """延迟加载视觉模型"""
        if self._vision_model_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("无法导入 transformers 库，请安装: pip install transformers")
        
        print(f"[VisualLanguageEncoder] 正在加载视觉模型: {self.model_path}")
        try:
            # 加载模型和processor
            self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.vision_processor_tokenizer = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False  # 避免警告，使用slow processor
            )
            self.vision_model.eval()  # 设置为评估模式
            self._vision_model_loaded = True
            print(f"[VisualLanguageEncoder] 视觉模型加载完成")
        except Exception as e:
            raise RuntimeError(f"加载视觉模型失败: {e}")
    
    def _get_cache_path(self) -> str:
        """获取特征缓存文件路径"""
        if self.dataset_name:
            cache_file = f"{self.dataset_name}_question_features.pt"
        else:
            cache_file = "question_features.pt"
        return os.path.join(self.cache_dir, cache_file)
    
    def _load_feature_cache(self):
        """加载特征缓存"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print(f"[VisualLanguageEncoder] 正在加载特征缓存: {cache_path}")
                self.question_features_cache = torch.load(
                    cache_path, 
                    map_location='cpu'  # 先加载到CPU，需要时再移到GPU
                )
                print(f"[VisualLanguageEncoder] 已加载 {len(self.question_features_cache)} 个题目特征")
            except Exception as e:
                print(f"[VisualLanguageEncoder] 警告: 加载缓存失败 {e}，将重新计算")
                self.question_features_cache = {}
        else:
            print(f"[VisualLanguageEncoder] 缓存文件不存在: {cache_path}")
            self.question_features_cache = {}
    
    def save_feature_cache(self):
        """保存特征缓存"""
        if not self.use_cache:
            print(f"[VisualLanguageEncoder] 缓存功能已禁用，跳过保存")
            return
        
        # 检查缓存是否为空（修复：空字典 len()=0，但我们应该检查是否真的为空）
        if len(self.question_features_cache) == 0:
            print(f"[VisualLanguageEncoder] 警告: 特征缓存为空，可能训练过程中未提取特征，跳过保存")
            print(f"[VisualLanguageEncoder] 提示: 这可能是正常的，如果特征提取失败或数据集较小")
            return
        
        cache_path = self._get_cache_path()
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        
        try:
            torch.save(self.question_features_cache, cache_path)
            print(f"[VisualLanguageEncoder] 已保存 {len(self.question_features_cache)} 个题目特征到 {cache_path}")
        except Exception as e:
            print(f"[VisualLanguageEncoder] 警告: 保存缓存失败 {e}")
    
    def encode_image(self, img_path: str, qid: Optional[int] = None) -> torch.Tensor:
        """
        编码单张图片
        
        Args:
            img_path: 图片路径
            qid: 问题ID（用于缓存）
            
        Returns:
            v_t: 题目特征向量 (d_question,)
        """
        # 检查缓存
        if qid is not None and qid in self.question_features_cache:
            feature = self.question_features_cache[qid].to(self.device)
            return feature
        
        # 加载视觉模型
        if not self._vision_model_loaded:
            self._load_vision_processor()
        
        # 提取视觉特征
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            
            # 准备输入
            prompt = "请分析这张图片中的题目内容，包括题干、选项和图形。"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理输入（使用 qwen_vl_utils 如果可用，否则手动处理）
            text = self.vision_processor_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if QWEN_VL_UTILS_AVAILABLE and process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                # 手动处理：只处理图片
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
            
            inputs = self.vision_processor_tokenizer(**processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取隐藏状态（使用最后几层的平均）
            with torch.no_grad():
                outputs = self.vision_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of tensors
                
                # 取最后2层的隐藏状态并平均池化
                # hidden_states[-1] 形状: (batch_size, seq_len, hidden_size)
                last_layers = hidden_states[-2:]  # 最后2层
                pooled_states = []
                for h in last_layers:
                    # 对序列维度求平均（排除padding）
                    mask = inputs.get('attention_mask', None)
                    if mask is not None:
                        # 使用attention mask进行masked mean
                        h_masked = h * mask.unsqueeze(-1)
                        pooled = h_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                    else:
                        pooled = h.mean(dim=1)
                    pooled_states.append(pooled)
                
                # 平均最后2层的池化结果
                hidden_state = torch.stack(pooled_states).mean(dim=0).squeeze(0)  # (hidden_size,)
            
            # 投影到目标维度（确保数据类型匹配）
            hidden_state = hidden_state.unsqueeze(0)  # (1, hidden_size)
            # 转换数据类型：如果模型是 bfloat16，转换为 float32 以匹配投影层
            if hidden_state.dtype == torch.bfloat16:
                hidden_state = hidden_state.float()
            v_t = self.feature_proj(hidden_state)  # (1, d_question)
            v_t = v_t.squeeze(0)  # (d_question,)
            
            # 缓存特征
            if qid is not None and self.use_cache:
                self.question_features_cache[qid] = v_t.detach().cpu()
            
            return v_t.to(self.device)
            
        except Exception as e:
            print(f"[VisualLanguageEncoder] 警告: 处理图片失败 {img_path}: {e}")
            import traceback
            traceback.print_exc()
            # 返回零向量作为fallback
            return torch.zeros(self.d_question, device=self.device)
    
    def encode_batch(
        self, 
        img_paths: List[str], 
        qids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        批量编码图片
        
        Args:
            img_paths: 图片路径列表
            qids: 问题ID列表（用于缓存）
            
        Returns:
            v_t: 题目特征矩阵 (batch_size, d_question)
        """
        batch_size = len(img_paths)
        features = []
        
        for i, img_path in enumerate(img_paths):
            qid = qids[i] if qids is not None else None
            feature = self.encode_image(img_path, qid)
            features.append(feature)
        
        return torch.stack(features)  # (batch_size, d_question)
    
    def predict_kc_distribution(self, v_t: torch.Tensor) -> torch.Tensor:
        """
        预测知识点分布
        
        Args:
            v_t: 题目特征 (batch_size, d_question) 或 (d_question,)
            
        Returns:
            k_t: 知识点分布 (batch_size, num_c) 或 (num_c,)
        """
        if v_t.dim() == 1:
            v_t = v_t.unsqueeze(0)  # (1, d_question)
        
        k_t = self.kc_classifier(v_t)  # (batch_size, num_c)
        
        if k_t.shape[0] == 1:
            k_t = k_t.squeeze(0)  # (num_c,)
        
        return k_t
    
    def forward(
        self, 
        qids: torch.Tensor,
        img_path_dict: Dict[int, str],
        return_kc: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            qids: 问题ID张量 (batch_size, seq_len)
            img_path_dict: 问题ID到图片路径的映射
            return_kc: 是否返回知识点分布
            
        Returns:
            v_t: 题目特征 (batch_size, seq_len, d_question)
            k_t: 知识点分布 (batch_size, seq_len, num_c) 或 None
        """
        batch_size, seq_len = qids.shape
        device = qids.device
        
        # 将qids移到CPU进行路径查找
        qids_cpu = qids.cpu().numpy()
        
        # 获取图片路径
        img_paths = []
        qids_list = []
        for b in range(batch_size):
            for s in range(seq_len):
                qid = int(qids_cpu[b, s])
                if qid in img_path_dict:
                    img_paths.append(img_path_dict[qid])
                    qids_list.append(qid)
                else:
                    # 如果找不到路径，使用占位符
                    img_paths.append(None)
                    qids_list.append(-1)
        
        # 批量编码（跳过None路径）
        valid_indices = [i for i, path in enumerate(img_paths) if path is not None]
        if valid_indices:
            valid_paths = [img_paths[i] for i in valid_indices]
            valid_qids = [qids_list[i] for i in valid_indices]
            valid_features = self.encode_batch(valid_paths, valid_qids)
        else:
            valid_features = torch.zeros((0, self.d_question), device=self.device)
        
        # 构建完整特征矩阵
        v_t = torch.zeros((batch_size, seq_len, self.d_question), device=device)
        valid_idx = 0
        for i, path in enumerate(img_paths):
            if path is not None:
                b, s = i // seq_len, i % seq_len
                v_t[b, s] = valid_features[valid_idx].to(device)
                valid_idx += 1
        
        # 预测知识点分布
        k_t = None
        if return_kc:
            v_t_flat = v_t.view(-1, self.d_question)  # (batch_size * seq_len, d_question)
            k_t_flat = self.predict_kc_distribution(v_t_flat)  # (batch_size * seq_len, num_c)
            k_t = k_t_flat.view(batch_size, seq_len, self.num_c)  # (batch_size, seq_len, num_c)
        
        return v_t, k_t
    
    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device
        
        # 移动分类头和投影层
        self.kc_classifier = self.kc_classifier.to(device)
        self.feature_proj = self.feature_proj.to(device)
        
        # 注意：视觉处理器保持原设备（通常已经通过device_map="auto"加载）
        
        return self


def build_img_path_dict(dataset_name: str, data_config: dict) -> Dict[int, str]:
    """
    构建问题ID到图片路径的映射字典
    
    Args:
        dataset_name: 数据集名称
        data_config: 数据配置字典
        
    Returns:
        img_path_dict: {qid: img_path}
    """
    img_path_dict = {}
    
    if dataset_name == "DBE_KT22":
        # DBE_KT22的图片路径
        dpath = data_config.get("dpath", "")
        # 需要找到实际的q_imgs目录
        # 通常路径为: data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/q_imgs
        possible_paths = [
            os.path.join(dpath, "../2_DBE_KT22_datafiles_100102_csv/q_imgs"),
            os.path.join(dpath, "../../2_DBE_KT22_datafiles_100102_csv/q_imgs"),
            "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/q_imgs"
        ]
        
        q_imgs_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                q_imgs_dir = path
                break
        
        if q_imgs_dir is None:
            print(f"[build_img_path_dict] 警告: 无法找到DBE_KT22的q_imgs目录")
            return img_path_dict
        
        # 扫描所有jpg文件
        for filename in os.listdir(q_imgs_dir):
            if filename.endswith('.jpg'):
                qid = int(filename.replace('.jpg', ''))
                img_path_dict[qid] = os.path.join(q_imgs_dir, filename)
        
    elif dataset_name == "XES3G5M":
        # XES3G5M的图片路径
        dpath = data_config.get("dpath", "")
        possible_paths = [
            os.path.join(dpath, "metadata/q_imgs"),
            "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/XES3G5M/metadata/q_imgs"
        ]
        
        q_imgs_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                q_imgs_dir = path
                break
        
        if q_imgs_dir is None:
            print(f"[build_img_path_dict] 警告: 无法找到XES3G5M的q_imgs目录")
            return img_path_dict
        
        # 扫描所有jpg文件
        for filename in os.listdir(q_imgs_dir):
            if filename.endswith('.jpg'):
                qid = int(filename.replace('.jpg', ''))
                img_path_dict[qid] = os.path.join(q_imgs_dir, filename)
    
    elif "NIPS_task34" in dataset_name or "nips_task34" in dataset_name or "Eedi" in dataset_name or "eedi" in dataset_name:
        # NIPS_task34的图片路径：图片直接存放在images文件夹下
        # 注意：NIPS_task34数据集本身就有题目图片，不需要预处理生成
        dpath = data_config.get("dpath", "")
        possible_paths = [
            os.path.join(dpath, "images"),
            os.path.join(dpath, "../images"),
            "/home3/zhiyu/code-5/CRKT/five-thinkkt/data/NIPS_task34/images",
            "/home3/zhiyu/code-5/CRKT/data/NIPS_task34/images"
        ]
        
        images_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                images_dir = path
                break
        
        if images_dir is None:
            print(f"[build_img_path_dict] 警告: 无法找到NIPS_task34的images目录，尝试过的路径: {possible_paths}")
            return img_path_dict
        
        # 尝试获取qid映射（如果存在，这是可选的）
        # 根据 prepare_q_imgs.py 的逻辑：
        # 1. 图片文件名（如 "823.jpg"）-> 原始qid（字符串 "823"）
        # 2. 如果有映射，使用 qid_ori2new[原始qid] 获取新qid
        # 3. 最终使用 int(新qid) 作为键
        qid_mapping = None
        try:
            # 方法1: 尝试从 keyid2idx.json 直接加载（更直接）
            keyid2idx_file = os.path.join(dpath, "keyid2idx.json")
            if os.path.exists(keyid2idx_file):
                import json
                with open(keyid2idx_file, 'r') as f:
                    keyid2idx = json.load(f)
                if "questions" in keyid2idx:
                    qid_mapping = keyid2idx["questions"]
                    print(f"[build_img_path_dict] NIPS_task34: 从 keyid2idx.json 加载了qid映射，映射数量: {len(qid_mapping)}")
        except Exception as e:
            pass
        
        # 方法2: 如果方法1失败，尝试使用 my_utils 函数（备选方案）
        if qid_mapping is None:
            try:
                import sys
                mapping_path = os.path.join(os.path.dirname(__file__), '../../../../prepare_q_img_for_kt_dataset')
                if os.path.exists(mapping_path):
                    sys.path.insert(0, mapping_path)
                    from my_utils import get_qid_ori2new
                    qid_mapping = get_qid_ori2new(data_config)
                    if qid_mapping:
                        print(f"[build_img_path_dict] NIPS_task34: 从 my_utils 加载了qid映射，映射数量: {len(qid_mapping)}")
            except (ImportError, ModuleNotFoundError):
                pass
            except Exception as e:
                pass
        
        # 扫描所有图片文件（支持多种格式）
        for filename in os.listdir(images_dir):
            # 支持.jpg, .jpeg, .png等格式
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 提取原始qid（文件名去掉扩展名，保持字符串格式以匹配映射键）
                qid_str = os.path.splitext(filename)[0]
                
                try:
                    # 根据 prepare_q_imgs.py 的逻辑：
                    # 如果有映射，使用映射后的qid；否则使用原始qid（转为int）
                    if qid_mapping and qid_str in qid_mapping:
                        # 使用映射后的qid（确保是int）
                        qid_new = qid_mapping[qid_str]
                        qid = int(qid_new) if isinstance(qid_new, (int, str)) else int(qid_str)
                    else:
                        # 没有映射，直接使用原始qid（转为int）
                        qid = int(qid_str)
                    
                    img_path = os.path.join(images_dir, filename)
                    img_path_dict[qid] = img_path
                except (ValueError, TypeError) as e:
                    print(f"[build_img_path_dict] 警告: 无法解析文件名的qid: {filename}, 错误: {e}")
                    continue
    
    print(f"[build_img_path_dict] 已构建 {len(img_path_dict)} 个问题ID到图片路径的映射")
    return img_path_dict

