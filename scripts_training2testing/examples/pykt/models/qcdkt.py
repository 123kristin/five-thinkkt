import json
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_config(dataset_name="XES3G5M"):
    """简洁的配置加载函数"""
    # 默认配置
    config = {
        'dataset_name': dataset_name,
        'num_q': 500, 'num_c': 100, 'max_concepts': 6,
        'dropout': 0.1, 'rnn_type': 'lstm', 'emb_type': 'qkcs',
        'dim_qc': 200, 'use_content_emb': True, 'content_dim': 1536,
        'content_type': 'text', 'trainable_content_emb': False,
        'use_analysis_emb': True, 'analysis_dim': 1536, 'analysis_type': 'generated',
        'trainable_analysis_emb': False, 'analysis_contrastive': True,
        'use_kc_emb': True, 'kc_dim': 1600, 'trainable_kc_emb': False,
        'contrastive_weight': 0.1, 'use_difficulty_contrastive': True,
        'difficulty_contrastive_weight': 0.05, 'difficulty_proj_dim': 64,
        'difficulty_temperature': 0.1
    }
    
    # 尝试从配置文件加载数据集信息
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'my_configs', 'data_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data_config = json.load(f)
            
            if dataset_name in data_config:
                info = data_config[dataset_name]
                config.update({
                    'num_q': info['num_q'],
                    'num_c': info['num_c'],
                    'max_concepts': info.get('max_concepts', 6)
                })
    except Exception as e:
        print(f"配置加载失败: {e}")
    
    return config


class CRKTNet(nn.Module):
    def __init__(self, config: dict):
        super(CRKTNet, self).__init__()
        self.model_name = 'crkt_net'
        
        # 优化设备选择逻辑，在初始化时确定设备
        self.device = self._determine_device()
        
        self.dropout = config.get('dropout', 0.1)
        self.num_q = config.get('num_q', 500)  # 问题数量
        self.num_c = config.get('num_c', 100)  # 知识点数量
        self.dim_qc = config.get('dim_qc', 200)  # 问题、知识点向量维度

        # 内容嵌入相关参数
        self.use_content_emb = config.get('use_content_emb', False)
        self.content_dim = config.get('content_dim', 512)
        self.content_type = config.get('content_type', 'text')
        self.trainable_content_emb = config.get('trainable_content_emb', False)
        self.gen_emb_path = config.get('gen_emb_path', '')
        self.dataset_name = config.get('dataset_name', '')
        
        # KC嵌入相关参数
        self.use_kc_emb = config.get('use_kc_emb', False)
        self.kc_dim = config.get('kc_dim', 1600)
        self.trainable_kc_emb = config.get('trainable_kc_emb', False)
        
        # 对比学习相关参数
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        self.use_analysis_emb = config.get('use_analysis_emb', False)
        self.analysis_dim = config.get('analysis_dim', 1536)
        self.analysis_type = config.get('analysis_type', 'generated')
        self.trainable_analysis_emb = config.get('trainable_analysis_emb', False)

        # 难度对比学习相关参数
        self.use_difficulty_contrastive = config.get('use_difficulty_contrastive', True)
        self.difficulty_contrastive_weight = config.get('difficulty_contrastive_weight', 0.05)
        self.difficulty_proj_dim = config.get('difficulty_proj_dim', 64)
        self.difficulty_temperature = config.get('difficulty_temperature', 0.1)
        
        # 解析嵌入使用模式参数
        self.analysis_contrastive = config.get('analysis_contrastive', True)
        
        # 确保布尔类型参数被正确转换
        self._convert_boolean_params()
        
        # 确保数值类型参数被正确转换
        self._convert_numeric_params()
        
        # 嵌入文件路径映射
        self.emb_file_mapping = {
            "content": "embedding_content.pkl",
            "content_image": "embedding_images_content.pkl",
            "generated": "embedding_generated_explanation.pkl",
            "original": "embedding_original_explanation.pkl",
            "kc": "embedding_kc.pkl",
            "kc_graph": "graph_embedding_kc.pkl"
        }
        
        # 嵌入数据
        self.content_emb_data = None
        self.kc_emb_data = None
        self.analysis_emb_data = None
        self.difficulty_labels = None

        # 根据是否使用KC嵌入决定知识点嵌入方式
        if self.use_kc_emb:
            self.KCEmbs = None
            self.kc_projection = None
        else:
            self.KCEmbs = nn.Embedding(self.num_c, self.dim_qc)
            self.kc_projection = None
        
        self.QEmbs = nn.Embedding(self.num_q, self.dim_qc)

        # 问题难度
        self.dim_difficulty = config.get('dim_difficulty', self.dim_qc // 2)

        # 能力模块
        self.dim_knowledge = self.dim_qc
        self.rnn_type = config.get('rnn_type', 'lstm')
        
        # 计算RNN输入维度
        rnn_input_dim = self.dim_qc * 4  # 基础维度：q_emb(d) + kc_avg(d) + zero_vector(2d)
        
        if self.use_content_emb:
            rnn_input_dim += self.dim_qc
        
        # RNN输入总维度: q_emb(d) + kc_avg(d) + zero_vector(2d) + content_emb(d)
        
        # 创建RNN层
        if self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(rnn_input_dim, self.dim_knowledge, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(rnn_input_dim, self.dim_knowledge, batch_first=True)

        # 增加2个MLP层
        # self.q_scores_extractor = nn.Sequential(
        #     nn.Linear(self.dim_knowledge, self.dim_knowledge),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_knowledge, self.dim_knowledge),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.dim_knowledge, self.num_q)
        # )
        # 原始输出
        self.q_scores_extractor = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_knowledge, self.num_q)
        )
        
        # 对比学习所需的投影层 - 延迟创建
        self.content_proj_layer = None
        self.analysis_proj_layer = None
        self.contrastive_proj = None
        
        # 根据数据集名称自动推断路径
        if not self.gen_emb_path and self.dataset_name:
            self.gen_emb_path = self._get_dataset_path(self.dataset_name)
        
        # 加载嵌入数据
        if self.gen_emb_path and (self.use_content_emb or self.use_analysis_emb or self.use_kc_emb):
            self._load_embeddings()
            self._create_projections()

        # 加载难度标签与创建难度投影层
        if self.use_difficulty_contrastive:
            self._load_difficulty_labels()
            self.difficulty_proj = nn.Linear(self.dim_qc, self.difficulty_proj_dim)
            self.difficulty_proj = self.difficulty_proj.to(self.device)

        # 将所有组件移动到正确设备
        # 兜底确保关键可选属性存在，避免后续 hasattr 判定之外的访问异常
        if not hasattr(self, 'content_projection'):
            self.content_projection = None
        if not hasattr(self, 'kc_projection'):
            self.kc_projection = None
        self._move_to_device()

    def _determine_device(self):
        """优化设备选择逻辑"""
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    device = torch.device("cuda:0")
            except ValueError:
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return device

    def _convert_boolean_params(self):
        """转换布尔类型参数"""
        if isinstance(self.use_content_emb, int):
            self.use_content_emb = bool(self.use_content_emb)
        if isinstance(self.use_analysis_emb, int):
            self.use_analysis_emb = bool(self.use_analysis_emb)
        if isinstance(self.use_kc_emb, int):
            self.use_kc_emb = bool(self.use_kc_emb)
        if isinstance(self.trainable_content_emb, int):
            self.trainable_content_emb = bool(self.trainable_content_emb)
        if isinstance(self.trainable_analysis_emb, int):
            self.trainable_analysis_emb = bool(self.trainable_analysis_emb)
        if isinstance(self.trainable_kc_emb, int):
            self.trainable_kc_emb = bool(self.trainable_kc_emb)
        if isinstance(self.use_difficulty_contrastive, int):
            self.use_difficulty_contrastive = bool(self.use_difficulty_contrastive)
        if isinstance(self.analysis_contrastive, int):
            self.analysis_contrastive = bool(self.analysis_contrastive)

    def _convert_numeric_params(self):
        """转换数值类型参数"""
        if isinstance(self.content_dim, str):
            try:
                self.content_dim = int(self.content_dim)
            except ValueError:
                self.content_dim = 512
                print(f"[CRKTNet] 警告: content_dim转换失败，使用默认值512")
        if isinstance(self.analysis_dim, str):
            try:
                self.analysis_dim = int(self.analysis_dim)
            except ValueError:
                self.analysis_dim = 1536
                print(f"[CRKTNet] 警告: analysis_dim转换失败，使用默认值1536")
        if isinstance(self.kc_dim, str):
            try:
                self.kc_dim = int(self.kc_dim)
            except ValueError:
                self.kc_dim = 1600
                print(f"[CRKTNet] 警告: kc_dim转换失败，使用默认值1600")
        if isinstance(self.difficulty_proj_dim, str):
            try:
                self.difficulty_proj_dim = int(self.difficulty_proj_dim)
            except ValueError:
                self.difficulty_proj_dim = 64
                print(f"[CRKTNet] 警告: difficulty_proj_dim转换失败，使用默认值64")
        
        # 确保contrastive_weight是float类型
        if isinstance(self.contrastive_weight, str):
            try:
                self.contrastive_weight = float(self.contrastive_weight)
            except ValueError:
                self.contrastive_weight = 0.1
                print(f"[CRKTNet] 警告: contrastive_weight转换失败，使用默认值0.1")
        if isinstance(self.difficulty_contrastive_weight, str):
            try:
                self.difficulty_contrastive_weight = float(self.difficulty_contrastive_weight)
            except ValueError:
                self.difficulty_contrastive_weight = 0.05
                print(f"[CRKTNet] 警告: difficulty_contrastive_weight转换失败，使用默认值0.05")
        if isinstance(self.difficulty_temperature, str):
            try:
                self.difficulty_temperature = float(self.difficulty_temperature)
            except ValueError:
                self.difficulty_temperature = 0.1
                print(f"[CRKTNet] 警告: difficulty_temperature转换失败，使用默认值0.1")

    def _move_to_device(self):
        """将所有组件移动到正确设备，减少重复移动"""
        self.QEmbs = self.QEmbs.to(self.device)
        if self.KCEmbs is not None:
            self.KCEmbs = self.KCEmbs.to(self.device)
        self.rnn_layer = self.rnn_layer.to(self.device)
        self.q_scores_extractor = self.q_scores_extractor.to(self.device)
        if hasattr(self, 'content_projection') and self.content_projection is not None:
            self.content_projection = self.content_projection.to(self.device)
        if hasattr(self, 'kc_projection') and self.kc_projection is not None:
            self.kc_projection = self.kc_projection.to(self.device)

    def _get_dataset_path(self, dataset_name):
        """根据数据集名称自动推断嵌入路径"""
        base_path = "/home3/zhiyu/code-4/kt_analysis_generation/data"
        
        if dataset_name == "XES3G5M":
            return os.path.join(base_path, "XES3G5M/generate_analysis/embeddings")
        elif dataset_name == "DBE_KT22":
            return os.path.join(base_path, "DBE_KT22/generate_analysis/embeddings")
        else:
            return ""
    
    def _get_kc_dataset_path(self):
        """根据数据集名称自动推断KC嵌入路径"""
        base_path = "/home3/zhiyu/code-4/kt_analysis_generation/data"
        
        if self.dataset_name == "XES3G5M":
            return os.path.join(base_path, "XES3G5M/generate_kc")
        elif self.dataset_name == "DBE_KT22":
            return os.path.join(base_path, "DBE_KT22/generate_kc")
        else:
            return ""

    def _get_difficulty_path(self):
        """根据数据集名称自动推断难度CSV路径"""
        base_path = "/home3/zhiyu/code-4/kt_analysis_generation/data"
        if self.dataset_name == "XES3G5M":
            return os.path.join(base_path, "XES3G5M/use_difficulty/gpt-4o_que_difficulty.csv")
        elif self.dataset_name == "DBE_KT22":
            return os.path.join(base_path, "DBE_KT22/use_difficulty/gpt-4o_que_difficulty.csv")
        else:
            return ""
    
    def _load_embeddings(self):
        """加载嵌入数据"""
        try:
            if self.use_content_emb:
                self._load_content_embedding()
            
            if self.use_analysis_emb:
                self._load_analysis_embedding()
            
            if self.use_kc_emb:
                self._load_kc_embedding()
                    
        except Exception as e:
            print(f"[CRKTNet] 加载嵌入异常: {e}")
            self.content_emb_data = None
            self.analysis_emb_data = None
            self.kc_emb_data = None
    
    def _create_projections(self):
        """在嵌入加载完成后创建投影层"""
        if self.use_content_emb and self.content_emb_data is not None:
            self._create_content_projection()
        
        # 移除解析嵌入主任务投影层的创建
        # if self.use_analysis_emb and self.analysis_emb_data is not None and self.analysis_main_task:
        #     self._create_analysis_projection()
        
        if self.use_kc_emb and self.kc_emb_data is not None:
            self._create_kc_projection()
        
        # 创建对比学习投影层（只有在有内容嵌入和解析嵌入数据，且解析嵌入参与对比学习时才创建）
        if (self.use_content_emb and self.content_emb_data is not None and 
            self.use_analysis_emb and self.analysis_emb_data is not None and 
            self.analysis_contrastive):
            self._create_contrastive_projections()
        
        # 确保对比学习投影层在正确的设备上
        if hasattr(self, 'content_proj_layer') and self.content_proj_layer is not None:
            if hasattr(self, 'device'):
                self.content_proj_layer = self.content_proj_layer.to(self.device)
                self.analysis_proj_layer = self.analysis_proj_layer.to(self.device)
                self.contrastive_proj = self.contrastive_proj.to(self.device)

        # 难度投影层确保在正确设备（若已创建）
        if hasattr(self, 'difficulty_proj') and self.difficulty_proj is not None and hasattr(self, 'device'):
            self.difficulty_proj = self.difficulty_proj.to(self.device)
    
    def _load_content_embedding(self):
        """加载内容嵌入"""
        try:
            # 根据content_type选择嵌入文件
            if self.content_type == "text":
                content_key = "content"
            elif self.content_type == "image":
                content_key = "content_image"
            else:
                print(f"[CRKTNet] 不支持的内容嵌入类型: {self.content_type}")
                self.content_emb_data = None
                return
                
            pkl_path = os.path.join(self.gen_emb_path, self.emb_file_mapping[content_key])
            
            if os.path.exists(pkl_path):
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    # 根据trainable_content_emb参数决定是否创建可训练嵌入
                    self.content_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_content_emb
                    )
                    if self.content_emb_data is not None:
                        # 更新content_dim为实际加载的维度
                        actual_content_dim = self.content_emb_data.shape[1]
                        if actual_content_dim != self.content_dim:
                            self.content_dim = actual_content_dim
                        return
            else:
                print(f"[CRKTNet] 内容嵌入文件不存在: {pkl_path}")
            
            self.content_emb_data = None
            
        except Exception as e:
            print(f"[CRKTNet] 加载内容嵌入异常: {e}")
            self.content_emb_data = None
    
    def _load_analysis_embedding(self):
        """加载解析嵌入"""
        try:
            # 根据analysis_type选择嵌入文件
            if self.analysis_type == "generated":
                analysis_key = "generated"
            elif self.analysis_type == "original":
                analysis_key = "original"
            else:
                print(f"[CRKTNet] 不支持的解析嵌入类型: {self.analysis_type}")
                self.analysis_emb_data = None
                return
                
            pkl_path = os.path.join(self.gen_emb_path, self.emb_file_mapping[analysis_key])
            
            if os.path.exists(pkl_path):
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    # 根据trainable_analysis_emb参数决定是否创建可训练嵌入
                    self.analysis_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_analysis_emb
                    )
                    if self.analysis_emb_data is not None:
                        # 更新analysis_dim为实际加载的维度
                        actual_analysis_dim = self.analysis_emb_data.shape[1]
                        if actual_analysis_dim != self.analysis_dim:
                            self.analysis_dim = actual_analysis_dim
                        return
            else:
                print(f"[CRKTNet] 解析嵌入文件不存在: {pkl_path}")
            
            self.analysis_emb_data = None
            
        except Exception as e:
            print(f"[CRKTNet] 加载解析嵌入异常: {e}")
            self.analysis_emb_data = None
    
    def _load_kc_embedding(self):
        """加载KC嵌入"""
        # 根据数据集名称推断KC嵌入路径
        kc_emb_path = self._get_kc_dataset_path()
        if not kc_emb_path:
            print(f"[CRKTNet] 无法推断KC嵌入路径，请检查数据集名称: {self.dataset_name}")
            self.kc_emb_data = None
            return
        
        # 优先尝试graph_embedding_kc.pkl，如果不存在则使用embedding_kc.pkl
        kc_files = ["kc", "kc_graph"]
        for kc_key in kc_files:
            pkl_path = os.path.join(kc_emb_path, self.emb_file_mapping[kc_key])
            if os.path.exists(pkl_path):
                emb_dict = self._load_pkl_embedding(pkl_path)
                if emb_dict is not None:
                    self.kc_emb_data = self._convert_dict_to_tensor(
                        emb_dict, make_trainable=self.trainable_kc_emb
                    )
                    if self.kc_emb_data is not None:
                        actual_kc_dim = self.kc_emb_data.shape[1]
                        if actual_kc_dim != self.kc_dim:
                            self.kc_dim = actual_kc_dim
                        return
            else:
                print(f"[CRKTNet] KC嵌入文件不存在: {pkl_path}")
        
        self.kc_emb_data = None
    
    def _create_content_projection(self):
        """在嵌入加载完成后创建内容投影层"""
        if self.use_content_emb and self.content_emb_data is not None:
            # 使用实际的嵌入维度创建投影层
            actual_content_dim = self.content_emb_data.shape[1]
            self.content_projection = nn.Linear(actual_content_dim, self.dim_qc)
            
            # 将投影层移动到正确的设备
            if hasattr(self, 'device'):
                self.content_projection = self.content_projection.to(self.device)
    
    def _create_analysis_projection(self):
        """在嵌入加载完成后创建解析投影层 - 已移除，不再需要"""
        # 这个方法不再需要，因为解析嵌入不参与主任务
        pass
    
    def _create_contrastive_projections(self):
        """创建对比学习所需的投影层"""
        if (self.use_content_emb and self.content_emb_data is not None and 
            self.use_analysis_emb and self.analysis_emb_data is not None):
            # 内容嵌入投影层
            self.content_proj_layer = nn.Linear(self.content_dim, self.dim_qc)
            # 解析嵌入投影层
            self.analysis_proj_layer = nn.Linear(self.analysis_dim, self.dim_qc)
            # 对比学习投影层
            self.contrastive_proj = nn.Linear(self.dim_qc, 128)
    
    def _create_kc_projection(self):
        """在嵌入加载完成后创建KC投影层"""
        if self.use_kc_emb and self.kc_emb_data is not None:
            # 使用实际的嵌入维度创建投影层
            actual_kc_dim = self.kc_emb_data.shape[1]
            self.kc_projection = nn.Linear(actual_kc_dim, self.dim_qc)
            
            # 将投影层移动到正确的设备
            if hasattr(self, 'device'):
                self.kc_projection = self.kc_projection.to(self.device)
    
    def _load_pkl_embedding(self, file_path):
        """加载pkl格式嵌入文件"""
        try:
            with open(file_path, 'rb') as f:
                emb_dict = pickle.load(f)
            return emb_dict
        except Exception as e:
            return None

    def _load_difficulty_labels(self):
        """加载难度CSV，构建按qid索引的难度张量（-1表示未知）"""
        try:
            csv_path = self._get_difficulty_path()
            if not csv_path or not os.path.exists(csv_path):
                # 即使没有难度CSV，也不报错，置空以跳过
                self.difficulty_labels = None
                return

            import csv
            label_tensor = torch.full((self.num_q,), -1, dtype=torch.long)
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        qid = int(row.get("qid", -1))
                        diff = int(row.get("difficulty", -1))
                        if 0 <= qid < self.num_q and diff >= 0:
                            # 将难度映射到从0开始的类别，假定CSV最小为1
                            mapped = max(0, diff - 1)
                            label_tensor[qid] = mapped
                    except Exception:
                        continue

            # 移动到设备
            if hasattr(self, 'device'):
                label_tensor = label_tensor.to(self.device)
            self.difficulty_labels = label_tensor
        except Exception as e:
            print(f"[CRKTNet] 加载难度CSV异常: {e}")
            self.difficulty_labels = None
    
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
                    print(f"[CRKTNet] 无法将max_qid转换为整数: {max_qid}")
                    return None
            
            # 获取嵌入维度
            sample_emb = next(iter(emb_dict.values()))
            if isinstance(sample_emb, list):
                emb_dim = len(sample_emb)
            elif hasattr(sample_emb, 'shape'):
                emb_dim = sample_emb.shape[-1]
            else:
                emb_dim = len(sample_emb)
            
            # 创建tensor并移动到正确设备
            # 确保tensor大小足够容纳所有qid
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
                    # 跳过这个嵌入，使用零向量
                    skipped_count += 1
                    continue
            
            # 如果指定为可训练，则转换为nn.Parameter
            if make_trainable:
                return nn.Parameter(emb_tensor, requires_grad=True)
            else:
                return emb_tensor
                
        except Exception as e:
            return None

    def get_content_emb(self, qids):
        """获取内容嵌入 - 优化版本"""
        if not self.use_content_emb or self.content_emb_data is None:
            return None
        
        device = qids.device
        batch_size, seq_len = qids.shape
        
        # 优化：只在需要时移动设备
        if self.content_emb_data.device != device:
            if isinstance(self.content_emb_data, nn.Parameter):
                self.content_emb_data.data = self.content_emb_data.data.to(device)
            else:
                self.content_emb_data = self.content_emb_data.to(device)
        
        valid_qids = torch.clamp(qids, 0, self.content_emb_data.size(0) - 1)
        
        # 优化内存分配策略
        if batch_size * seq_len > 10000:
            content_emb = torch.zeros(batch_size, seq_len, self.content_emb_data.size(1), 
                                    device=device, dtype=self.content_emb_data.dtype)
            chunk_size = min(1000, batch_size * seq_len // 10)  # 动态调整chunk大小
            for i in range(0, batch_size * seq_len, chunk_size):
                end_idx = min(i + chunk_size, batch_size * seq_len)
                chunk_qids = valid_qids.view(-1)[i:end_idx]
                chunk_emb = self.content_emb_data[chunk_qids]
                content_emb.view(-1, self.content_emb_data.size(1))[i:end_idx] = chunk_emb
        else:
            content_emb = self.content_emb_data[valid_qids]
        
        # 数值稳定性检查
        if torch.isnan(content_emb).any() or torch.isinf(content_emb).any():
            print(f"[CRKTNet] 警告: content_emb包含NaN或Inf值")
            content_emb = torch.zeros_like(content_emb)
        
        return content_emb

    def get_analysis_emb(self, qids):
        """获取解析嵌入 - 优化版本"""
        if not self.use_analysis_emb or self.analysis_emb_data is None:
            return None
        
        device = qids.device
        batch_size, seq_len = qids.shape
        
        # 优化：只在需要时移动设备
        if self.analysis_emb_data.device != device:
            if isinstance(self.analysis_emb_data, nn.Parameter):
                self.analysis_emb_data.data = self.analysis_emb_data.data.to(device)
            else:
                self.analysis_emb_data = self.analysis_emb_data.to(device)
        
        valid_qids = torch.clamp(qids, 0, self.analysis_emb_data.size(0) - 1)
        
        # 优化内存分配策略
        if batch_size * seq_len > 10000:
            analysis_emb = torch.zeros(batch_size, seq_len, self.analysis_emb_data.size(1), 
                                     device=device, dtype=self.analysis_emb_data.dtype)
            chunk_size = min(1000, batch_size * seq_len // 10)  # 动态调整chunk大小
            for i in range(0, batch_size * seq_len, chunk_size):
                end_idx = min(i + chunk_size, batch_size * seq_len)
                chunk_qids = valid_qids.view(-1)[i:end_idx]
                chunk_emb = self.analysis_emb_data[chunk_qids]
                analysis_emb.view(-1, self.analysis_emb_data.size(1))[i:end_idx] = chunk_emb
        else:
            analysis_emb = self.analysis_emb_data[valid_qids]
        
        # 数值稳定性检查
        if torch.isnan(analysis_emb).any() or torch.isinf(analysis_emb).any():
            print(f"[CRKTNet] 警告: analysis_emb包含NaN或Inf值")
            analysis_emb = torch.zeros_like(analysis_emb)
        
        return analysis_emb

    def get_kc_emb(self, c_ids):
        """获取KC嵌入 - 优化版本"""
        if not self.use_kc_emb or self.kc_emb_data is None:
            return None
        
        device = c_ids.device
        batch_size, seq_len, max_concepts = c_ids.shape
        
        # 优化：只在需要时移动设备
        if self.kc_emb_data.device != device:
            if isinstance(self.kc_emb_data, nn.Parameter):
                self.kc_emb_data.data = self.kc_emb_data.data.to(device)
            else:
                self.kc_emb_data = self.kc_emb_data.to(device)
        
        valid_c_ids = torch.clamp(c_ids, 0, self.kc_emb_data.size(0) - 1)
        
        # 优化内存分配策略
        if batch_size * seq_len * max_concepts > 10000:
            kc_emb = torch.zeros(batch_size, seq_len, max_concepts, self.kc_emb_data.size(1), 
                                device=device, dtype=self.kc_emb_data.dtype)
            chunk_size = min(1000, batch_size * seq_len * max_concepts // 10)  # 动态调整chunk大小
            total_elements = batch_size * seq_len * max_concepts
            for i in range(0, total_elements, chunk_size):
                end_idx = min(i + chunk_size, total_elements)
                chunk_c_ids = valid_c_ids.view(-1)[i:end_idx]
                chunk_emb = self.kc_emb_data[chunk_c_ids]
                kc_emb.view(-1, self.kc_emb_data.size(1))[i:end_idx] = chunk_emb
        else:
            kc_emb = self.kc_emb_data[valid_c_ids]
        
        # 数值稳定性检查
        if torch.isnan(kc_emb).any() or torch.isinf(kc_emb).any():
            print(f"[CRKTNet] 警告: kc_emb包含NaN或Inf值")
            kc_emb = torch.zeros_like(kc_emb)
        
        return kc_emb

    def to(self, device):
        """重写to方法，优化设备移动逻辑"""
        if device == self.device:
            return self  # 如果已经是目标设备，直接返回
        
        super().to(device)
        self.device = device
        
        # 移动所有模型组件到新设备
        self.QEmbs = self.QEmbs.to(device)
        if self.KCEmbs is not None:
            self.KCEmbs = self.KCEmbs.to(device)
        self.rnn_layer = self.rnn_layer.to(device)
        self.q_scores_extractor = self.q_scores_extractor.to(device)
        
        # 移动投影层
        if hasattr(self, 'content_projection') and self.content_projection is not None:
            self.content_projection = self.content_projection.to(device)
        if hasattr(self, 'kc_projection') and self.kc_projection is not None:
            self.kc_projection = self.kc_projection.to(device)
        if hasattr(self, 'difficulty_proj') and self.difficulty_proj is not None:
            self.difficulty_proj = self.difficulty_proj.to(device)
        
        # 移动对比学习投影层
        if self.content_proj_layer is not None:
            self.content_proj_layer = self.content_proj_layer.to(device)
        if self.analysis_proj_layer is not None:
            self.analysis_proj_layer = self.analysis_proj_layer.to(device)
        if self.contrastive_proj is not None:
            self.contrastive_proj = self.contrastive_proj.to(device)
        
        # 移动嵌入数据（如果是nn.Parameter）
        if self.content_emb_data is not None and isinstance(self.content_emb_data, nn.Parameter):
            self.content_emb_data.data = self.content_emb_data.data.to(device)
        elif self.content_emb_data is not None:
            self.content_emb_data = self.content_emb_data.to(device)
            
        if self.analysis_emb_data is not None and isinstance(self.analysis_emb_data, nn.Parameter):
            self.analysis_emb_data.data = self.analysis_emb_data.data.to(device)
        elif self.analysis_emb_data is not None:
            self.analysis_emb_data = self.analysis_emb_data.to(device)
            
        if self.kc_emb_data is not None and isinstance(self.kc_emb_data, nn.Parameter):
            self.kc_emb_data.data = self.kc_emb_data.data.to(device)
        elif self.kc_emb_data is not None:
            self.kc_emb_data = self.kc_emb_data.to(device)

        # 移动难度标签
        if self.difficulty_labels is not None:
            self.difficulty_labels = self.difficulty_labels.to(device)
        
        print(f"[CRKTNet] 模型已移动到设备: {device}")
        return self

    def get_kc_avg_emb(self, c, pad_idx=-1):
        # 1. 掩码：True 表示有效索引
        mask = c != pad_idx  # [bz, len, max_concepts]

        # 2. 安全索引：把 -1 等填充值映射到 0（或其他任意合法 id，后面会被 mask 忽略）
        c_safe = c.masked_fill(~mask, 0)  # [bz, len, max_concepts]

        if self.use_kc_emb and self.kc_emb_data is not None:
            # 使用KC嵌入
            # 3. 查表得到所有向量；填充位置向量将被后续 mask 忽略
            kc_embs = self.get_kc_emb(c_safe)  # [bz, len, max_concepts, kc_emb_dim]
            
            # 4. 将填充位置向量置 0
            kc_embs = kc_embs * mask.unsqueeze(-1)  # [bz, len, max_concepts, kc_emb_dim]
            
            # 5. 求均值（避免除 0）
            sum_emb = kc_embs.sum(dim=-2)  # [bz, len, kc_emb_dim]
            valid_cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1) # [bz, len, 1]
            mean_emb = sum_emb / valid_cnt  # [bz, len, kc_emb_dim]
            
            # 6. 通过投影层降维到目标维度
            if self.kc_projection is not None:
                mean_emb = self.kc_projection(mean_emb)  # [bz, len, dim_qc]
            
            return mean_emb
        else:
            # 使用随机初始化的知识点嵌入
            # 3. 查表得到所有向量；填充位置向量将被后续 mask 忽略
            embs = self.KCEmbs(c_safe)  # [bz, len, max_concepts, emb_size]

            # 4. 将填充位置向量置 0
            embs = embs * mask.unsqueeze(-1)  # [bz, len, max_concepts, emb_size]

            # 5. 求均值（避免除 0）
            sum_emb = embs.sum(dim=-2)  # [bz, len, emb_size]
            valid_cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1) # [bz, len, 1]
            mean_emb = sum_emb / valid_cnt  # [bz, len, emb_size]

            return mean_emb

    def forward(self, q, c, r, q_shift, return_all=False, training=True):
        """
        :param q: (bz, interactions_seq_len - 1)
            the first (interaction_seq_len - 1) q in an interaction sequence of a student
        :param c: (bz, interactions_seq_len - 1, max_concepts)
        :param r: (bz, interactions_seq_len - 1)
            the first (interaction_seq_len - 1) responses in an interaction sequence of a student
        :param q_shift: (bz, interactions_seq_len - 1)
            the last (interaction_seq_len - 1) q  in an interaction sequence of a student
        :param return_all: 是否返回所有输出
        :param training: 是否处于训练模式，只有在训练模式下才计算对比损失

        :return: (bz, interaction_seq_len - 1)
            the predicted (interaction_seq_len - 1) responses
        """
        bz, num_interactions = q.shape  # num_interactions = interactions_seq_len - 1

        # 移入 device - 优化：只在需要时移动
        if q.device != self.device:
            q = q.to(self.device)
        if c.device != self.device:
            c = c.to(self.device)
        if r.device != self.device:
            r = r.to(self.device)
        if q_shift.device != self.device:
            q_shift = q_shift.to(self.device)

        q_emb = self.QEmbs(q)  # [bz, num_interactions, dim_qc]
        
        # 获取内容嵌入（如果启用）
        content_emb = None
        if self.use_content_emb and self.content_projection is not None:
            content_emb = self.get_content_emb(q)
            if content_emb is not None:
                content_emb = self.content_projection(content_emb)  # [bz, num_interactions, dim_qc]
        
        # 获取KC平均嵌入 - 使用实际的输入c，这是正确的
        kc_avg_embs = self.get_kc_avg_emb(c)  # (bz, num_interactions, dim_qc)
        
        # 构建输入嵌入
        input_components = [q_emb]  # 问题嵌入总是包含
        
        if self.use_content_emb and content_emb is not None:
            input_components.append(content_emb)  # 内容嵌入
        
        input_components.append(kc_avg_embs)  # KC嵌入总是包含
        
        # 零向量固定为2*dim_qc
        zero_vector = torch.zeros(bz, num_interactions, self.dim_qc * 2, device=self.device)
        
        # 构建答对/答错时的输入
        correct_input = torch.cat(input_components + [zero_vector], dim=-1)   # [bz, num_interactions, F]
        wrong_input   = torch.cat([zero_vector] + input_components, dim=-1)   # [bz, num_interactions, F]
        
        # 使用与输入同形状的布尔掩码进行选择
        mask = (r == 1).unsqueeze(-1).expand_as(correct_input)  # [bz, num_interactions, F]
        e_emb = torch.where(mask, correct_input, wrong_input)

        lstms_out, _ = self.rnn_layer(e_emb)  # (bz, interactions_seq_len - 1, dim_knowledge)
        q_scores = self.q_scores_extractor(lstms_out)  # (bz, num_interactions, num_q)
        q_scores = torch.sigmoid(q_scores) # (bz, num_interactions, num_q)
        
        # 计算解决该问题的能力
        y = (q_scores * F.one_hot(q_shift.long(), self.num_q)).sum(-1)  # (bz, num_interactions)

        # 只有在训练模式下才计算对比学习损失
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if training and (self.contrastive_weight > 0 and 
            self.use_content_emb and self.use_analysis_emb and self.analysis_contrastive):
            
            raw_content_emb = self.get_content_emb(q)
            raw_analysis_emb = self.get_analysis_emb(q)
            
            if raw_content_emb is not None and raw_analysis_emb is not None:
                contrastive_loss = self._compute_contrastive_loss(raw_content_emb, raw_analysis_emb)

        # 只有在训练模式下才计算难度对比学习损失
        difficulty_loss = torch.tensor(0.0, device=self.device)
        if training and self.use_difficulty_contrastive and self.difficulty_contrastive_weight > 0:
            try:
                difficulty_loss = self._compute_difficulty_contrastive_loss(q, temperature=self.difficulty_temperature)
            except Exception as e:
                print(f"[CRKTNet] 计算难度对比损失时出错: {e}")
                difficulty_loss = torch.tensor(0.0, device=self.device)

        if return_all:
            return y, q_scores, contrastive_loss, difficulty_loss
        else:
            return y, contrastive_loss, difficulty_loss
    
    def _compute_contrastive_loss(self, content_embs, analysis_embs, temperature=0.1):
        """
        计算标准InfoNCE对比学习损失
        """
        # 如果对比学习权重小于等于0，或者没有必要的组件，则返回0
        if (self.contrastive_weight <= 0 or 
            not hasattr(self, 'content_proj_layer') or self.content_proj_layer is None or
            not hasattr(self, 'content_proj_layer') or self.analysis_proj_layer is None or
            not hasattr(self, 'contrastive_proj') or self.contrastive_proj is None or
            content_embs is None or analysis_embs is None):
            return torch.tensor(0.0, device=self.device)

        try:
            # 获取投影后的特征
            content_proj_input = self.content_proj_layer(content_embs)
            analysis_proj_input = self.analysis_proj_layer(analysis_embs)
            
            # 进一步投影到对比学习空间
            content_proj = self.contrastive_proj(content_proj_input)  # (batch_size, seq_len, 128)
            analysis_proj = self.contrastive_proj(analysis_proj_input)  # (batch_size, seq_len, 128)
            
            # 对序列维度进行平均池化，得到每个样本的表示
            content_features = content_proj.mean(dim=1)  # (batch_size, 128)
            analysis_features = analysis_proj.mean(dim=1)  # (batch_size, 128)
            
            # L2归一化
            content_features = F.normalize(content_features, p=2, dim=1)
            analysis_features = F.normalize(analysis_features, p=2, dim=1)
            
            # 使用标准InfoNCE对比损失
            # 计算相似性矩阵
            similarity_matrix = torch.matmul(content_features, analysis_features.T) / temperature
            
            # 创建正样本标签（对角线为1）
            batch_size = similarity_matrix.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
            
            # 计算InfoNCE损失
            # 对每个content，其对应的analysis是正样本，其他是负样本
            content_loss = F.cross_entropy(similarity_matrix, labels)
            
            # 对每个analysis，其对应的content是正样本，其他是负样本
            analysis_loss = F.cross_entropy(similarity_matrix.T, labels)
            
            # 总的对比学习损失
            contrastive_loss = (content_loss + analysis_loss) / 2
            
            return contrastive_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)

    def _compute_difficulty_contrastive_loss(self, qids, temperature=0.1):
        """基于难度标签的监督式对比学习损失。
        仅使用当前batch内的题目ID与其嵌入，避免任何数据泄露（不跨batch构造样本）。
        """
        if (self.difficulty_labels is None or self.difficulty_proj is None):
            return torch.tensor(0.0, device=self.device)

        # 取题目嵌入并投影
        q_emb = self.QEmbs(qids)  # [bz, T, d]
        feats = self.difficulty_proj(q_emb)  # [bz, T, d_proj]
        feats = F.normalize(feats, p=2, dim=-1)

        # 拉平成 [N, d_proj]
        bz, T, d_proj = feats.shape
        N = bz * T
        feats = feats.reshape(N, d_proj)

        # 对应的难度标签 [N]
        labels = self.difficulty_labels[qids].reshape(-1)  # [N]

        # 过滤未知难度（<0）
        valid_mask = labels >= 0
        if valid_mask.sum() <= 1:
            return torch.tensor(0.0, device=self.device)

        feats_valid = feats[valid_mask]
        labels_valid = labels[valid_mask]
        M = feats_valid.size(0)

        # 相似度矩阵与温度缩放
        sim = torch.matmul(feats_valid, feats_valid.T) / temperature  # [M, M]

        # 去除自对比（对角）
        diag_mask = torch.eye(M, device=self.device).bool()
        sim = sim.masked_fill(diag_mask, float('-inf'))

        # 构造正样本掩码（同难度）
        labels_row = labels_valid.unsqueeze(1)  # [M,1]
        labels_col = labels_valid.unsqueeze(0)  # [1,M]
        pos_mask = (labels_row == labels_col) & (~diag_mask)

        # 对每一行：
        #   denom = logsumexp(sim[i, :])
        #   pos   = logsumexp(sim[i, positives])
        #   loss_i = -(pos - denom)
        # 若该行无任何正样本，则跳过。
        log_denom = torch.logsumexp(sim, dim=1)  # [M]

        # 为避免空正样本导致的 -inf，先用一个极小mask
        neg_inf = torch.full_like(sim, float('-inf'))
        pos_only = torch.where(pos_mask, sim, neg_inf)
        log_pos = torch.logsumexp(pos_only, dim=1)  # [M]

        # 有正样本的行
        has_pos = pos_mask.any(dim=1)
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        loss_vec = -(log_pos[has_pos] - log_denom[has_pos])
        return loss_vec.mean() if loss_vec.numel() > 0 else torch.tensor(0.0, device=self.device)


class CRKT(nn.Module):
    """
    消融实验用，没有多头注意力的block 块，即没有建模q_kcs的权重以及
    """

    def __init__(self, config):
        super(CRKT, self).__init__()
        self.model_name = 'crkt'
        self.emb_type = config.get('emb_type', 'qkcs')
        
        # 优化设备选择逻辑，在初始化时确定设备
        self.device = self._determine_device()
        
        self.model = CRKTNet(config)
        
        # 确保模型在正确设备上
        if self.model.device != self.device:
            self.model = self.model.to(self.device)

    def _determine_device(self):
        """优化设备选择逻辑"""
        if torch.cuda.is_available():
            current_gpu_id = os.environ.get('CURRENT_GPU_ID', '0')
            try:
                gpu_id = int(current_gpu_id)
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    device = torch.device("cuda:0")
            except ValueError:
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return device

    def train_one_step(self, data):
        # 前向返回：y, contrastive_loss(内容×解析), difficulty_loss(难度对比)
        # 训练时传递training=True，确保计算对比损失
        y, contrastive_loss, difficulty_loss = self.model(
            data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'], training=True
        )

        # 优化：只在需要时移动设备
        sm = data['smasks']
        r_shift = data['shft_rseqs']
        
        if sm.device != self.device:
            sm = sm.to(self.device)
        if r_shift.device != self.device:
            r_shift = r_shift.to(self.device)
        
        # calculate main loss
        main_loss = self.get_loss(y, r_shift, sm)
        
        # total loss = 主任务 + 内容×解析对比 + 难度对比
        total_loss = (
            main_loss
            + self.model.contrastive_weight * contrastive_loss
            + self.model.difficulty_contrastive_weight * difficulty_loss
        )

        return y, total_loss, main_loss, contrastive_loss

    def predict_one_step(self, data):
        # 预测时传递training=False，避免计算对比损失
        y, _, _ = self.model(data['qseqs'], data['cseqs'], data['rseqs'], data['shft_qseqs'], training=False)
        return y

    def get_loss(self, ys, rshft, sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        
        loss = F.binary_cross_entropy(y_pred.double(), y_true.double())
        return loss

    def to(self, device):
        """重写to方法，优化设备移动逻辑"""
        if device == self.device:
            return self  # 如果已经是目标设备，直接返回
        
        super().to(device)
        self.device = device
        self.model.to(device)
        return self

if __name__ == '__main__':
    """简洁的模型测试"""
    
    def test_model(config):
        try:
            model = CRKT(config)
            print(f"✅ {config['dataset_name']}: 模型创建成功")
            
            # 简单测试
            bz, seq_len = 2, 10
            q = torch.randint(0, config['num_q'], (bz, seq_len - 1))
            c = torch.randint(0, config['num_q'], (bz, seq_len - 1, config['max_concepts']))
            r = torch.randint(0, 2, (bz, seq_len - 1))
            q_shift = torch.randint(0, config['num_q'], (bz, seq_len - 1))
            
            with torch.no_grad():
                y, _, _ = model.model(q, c, r, q_shift)
                print(f"✅ {config['dataset_name']}: 测试通过")
                
        except Exception as e:
            print(f"❌ {config['dataset_name']}: 测试失败 - {e}")
    
    # 测试数据集
    for dataset in ["XES3G5M", "DBE_KT22"]:
        config = load_config(dataset)
        test_model(config)
    
    print("\n💡 使用: python wandb_crkt_train.py --dataset_name XES3G5M --gpu_id 0")
