import pandas as pd
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import os
import numpy as np
import csv
from .data_loader import KTDataset

class CzyKTDataset(KTDataset):
    """
    CzyKT专用数据集类，继承KTDataset并添加difficulty处理功能
    """
    def __init__(self, file_path, input_type, folds, qtest=False, difficulty_file_path=None, gen_difficulty_file_path=None):
        """
        Args:
            file_path: 训练/测试文件路径
            input_type: 输入类型列表，如["questions", "concepts"]
            folds: 用于生成数据集的fold集合，-1表示测试数据
            qtest: 是否为问题级别评估
            difficulty_file_path: global difficulty CSV文件路径
            gen_difficulty_file_path: generated difficulty CSV文件路径
        """
        self.difficulty_file_path = difficulty_file_path
        self.gen_difficulty_file_path = gen_difficulty_file_path
        self.qid_to_difficulty = {}
        self.qid_to_gen_difficulty = {}
        
        # 加载global difficulty映射
        if difficulty_file_path and os.path.exists(difficulty_file_path):
            self._load_difficulty_mapping()
        # 加载generated difficulty映射
        if gen_difficulty_file_path and os.path.exists(gen_difficulty_file_path):
            self._load_gen_difficulty_mapping()
        
        # 调用父类初始化
        super().__init__(file_path, input_type, folds, qtest)
        
        # 添加difficulty序列到已有数据中
        self._add_difficulty_sequences()
        # 添加generated difficulty序列
        self._add_gen_difficulty_sequences()
    
    def _load_difficulty_mapping(self):
        """加载global difficulty CSV文件，建立qid→difficulty映射"""
        try:
            df = pd.read_csv(self.difficulty_file_path)
            for _, row in df.iterrows():
                qid = int(row['qid'])
                original_difficulty = int(row['difficulty'])
                converted_difficulty = max(0, original_difficulty - 1)
                self.qid_to_difficulty[qid] = converted_difficulty
        except Exception as e:
            self.qid_to_difficulty = {}
    
    def _load_gen_difficulty_mapping(self):
        """加载generated difficulty CSV文件，建立qid→gen_difficulty映射（不做-1偏移，直接1/2/3）"""
        try:
            df = pd.read_csv(self.gen_difficulty_file_path)
            for _, row in df.iterrows():
                qid = int(row['qid'])
                gen_difficulty = int(row['difficulty'])  # 保持1/2/3
                self.qid_to_gen_difficulty[qid] = gen_difficulty
        except Exception as e:
            self.qid_to_gen_difficulty = {}
    
    def _add_difficulty_sequences(self):
        """为已加载的数据添加difficulty序列（qdseqs）和交互嵌入专用difficulty序列（difficulty_seq_for_interaction）"""
        if not hasattr(self, 'dori') or not self.qid_to_difficulty:
            return
        # 为每个用户序列生成difficulty序列
        qdseqs = []
        difficulty_seq_for_interaction = []
        for i in range(len(self.dori['qseqs'])):
            qseq = self.dori['qseqs'][i]
            qdseq = []
            diff_inter = []
            for qid in qseq:
                qid_val = int(qid.item()) if hasattr(qid, 'item') else int(qid)
                if qid_val == -1:  # padding值
                    qdseq.append(-1)
                    diff_inter.append(-1)
                else:
                    # 查找difficulty，如果不存在则默认为0（最低难度）
                    difficulty = self.qid_to_difficulty.get(qid_val, 0)
                    qdseq.append(difficulty)
                    diff_inter.append(difficulty)
            qdseqs.append(qdseq)
            difficulty_seq_for_interaction.append(diff_inter)
        qdseqs_tensor = LongTensor(qdseqs)
        difficulty_seq_for_interaction_tensor = LongTensor(difficulty_seq_for_interaction)
        if len(self.dori['qseqs']) > 0:
            reference_device = self.dori['qseqs'].device
            qdseqs_tensor = qdseqs_tensor.to(reference_device)
            difficulty_seq_for_interaction_tensor = difficulty_seq_for_interaction_tensor.to(reference_device)
        self.dori['qdseqs'] = qdseqs_tensor
        self.dori['difficulty_seq_for_interaction'] = difficulty_seq_for_interaction_tensor
    
    def _add_gen_difficulty_sequences(self):
        """为已加载的数据添加generated difficulty序列"""
        if not hasattr(self, 'dori') or not self.qid_to_gen_difficulty:
            return
        gen_qdseqs = []
        for i in range(len(self.dori['qseqs'])):
            qseq = self.dori['qseqs'][i]
            gen_qdseq = []
            for qid in qseq:
                qid_val = int(qid.item()) if hasattr(qid, 'item') else int(qid)
                if qid_val == -1:
                    gen_qdseq.append(-1)
                else:
                    gen_difficulty = self.qid_to_gen_difficulty.get(qid_val, 1)  # 默认1
                    gen_qdseq.append(gen_difficulty)
            gen_qdseqs.append(gen_qdseq)
        gen_qdseqs_tensor = LongTensor(gen_qdseqs)
        if len(self.dori['qseqs']) > 0:
            reference_device = self.dori['qseqs'].device
            gen_qdseqs_tensor = gen_qdseqs_tensor.to(reference_device)
        self.dori['gen_difficulty_seqs'] = gen_qdseqs_tensor
    
    def __getitem__(self, index):
        """
        获取单个样本，在父类基础上添加difficulty信息
        """
        # 调用父类方法获取基础数据
        if self.qtest:
            dcur, dqtest = super().__getitem__(index)
        else:
            dcur = super().__getitem__(index)
        
        # 添加global difficulty序列
        if hasattr(self, 'dori') and 'qdseqs' in self.dori:
            mseqs = dcur["masks"]
            qdseqs_raw = self.dori["qdseqs"][index][:-1]
            qdshft_seqs_raw = self.dori["qdseqs"][index][1:]
            if qdseqs_raw.device != mseqs.device:
                qdseqs_raw = qdseqs_raw.to(mseqs.device)
                qdshft_seqs_raw = qdshft_seqs_raw.to(mseqs.device)
            qdseqs = qdseqs_raw * mseqs
            qdshft_seqs = qdshft_seqs_raw * mseqs
            dcur["qdseqs"] = qdseqs
            dcur["shft_qdseqs"] = qdshft_seqs
        # 添加交互嵌入专用difficulty序列
        if hasattr(self, 'dori') and 'difficulty_seq_for_interaction' in self.dori:
            mseqs = dcur["masks"]
            diff_inter_raw = self.dori["difficulty_seq_for_interaction"][index][:-1]
            diff_inter_shft_raw = self.dori["difficulty_seq_for_interaction"][index][1:]
            if diff_inter_raw.device != mseqs.device:
                diff_inter_raw = diff_inter_raw.to(mseqs.device)
                diff_inter_shft_raw = diff_inter_shft_raw.to(mseqs.device)
            diff_inter = diff_inter_raw * mseqs
            diff_inter_shft = diff_inter_shft_raw * mseqs
            dcur["difficulty_seq_for_interaction"] = diff_inter
            dcur["shft_difficulty_seq_for_interaction"] = diff_inter_shft
        # 添加generated difficulty序列
        if hasattr(self, 'dori') and 'gen_difficulty_seqs' in self.dori:
            mseqs = dcur["masks"]
            gen_qdseqs_raw = self.dori["gen_difficulty_seqs"][index][:-1]
            gen_qdshft_seqs_raw = self.dori["gen_difficulty_seqs"][index][1:]
            if gen_qdseqs_raw.device != mseqs.device:
                gen_qdseqs_raw = gen_qdseqs_raw.to(mseqs.device)
                gen_qdshft_seqs_raw = gen_qdshft_seqs_raw.to(mseqs.device)
            gen_qdseqs = gen_qdseqs_raw * mseqs
            gen_qdshft_seqs = gen_qdshft_seqs_raw * mseqs
            dcur["gen_difficulty_seqs"] = gen_qdseqs
            dcur["shft_gen_difficulty_seqs"] = gen_qdshft_seqs
        if self.qtest:
            return dcur, dqtest
        else:
            return dcur 