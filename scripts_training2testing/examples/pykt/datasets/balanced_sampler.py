#!/usr/bin/env python3
"""
平衡采样器，确保每个batch包含所有难度类别
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict


class DifficultyBalancedSampler(Sampler):
    """
    难度平衡采样器
    确保每个batch都包含所有difficulty类别的样本
    """
    
    def __init__(self, dataset, batch_size, num_difficulty_classes=3, min_samples_per_class=2):
        """
        Args:
            dataset: CzyKTDataset实例
            batch_size: batch大小 
            num_difficulty_classes: 难度类别数量 (0,1,2)
            min_samples_per_class: 每个类别在batch中的最小样本数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_difficulty_classes = num_difficulty_classes
        self.min_samples_per_class = min_samples_per_class
        
        # 按难度类别分组样本索引
        self.difficulty_to_indices = self._group_by_difficulty()
        
        # 计算每个难度类别的样本数
        self.class_counts = {diff: len(indices) for diff, indices in self.difficulty_to_indices.items()}
        print(f"📊 BalancedSampler难度分布: {self.class_counts}")
        
        # 计算总的batch数量
        total_samples = len(dataset)
        self.num_batches = total_samples // batch_size
        
    def _group_by_difficulty(self):
        """按难度类别分组样本索引"""
        difficulty_to_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if 'qdseqs' in sample:
                    # 获取样本的聚合难度
                    qdseqs = sample['qdseqs']
                    valid_difficulties = qdseqs[qdseqs != -1]
                    if len(valid_difficulties) > 0:
                        # 使用相同的聚合策略
                        mean_diff = valid_difficulties.float().mean().round().long().clamp(0, 2).item()
                        difficulty_to_indices[mean_diff].append(idx)
                    else:
                        difficulty_to_indices[0].append(idx)  # 默认为easy
                else:
                    difficulty_to_indices[0].append(idx)  # 默认为easy
            except:
                difficulty_to_indices[0].append(idx)  # 出错时默认为easy
        
        return difficulty_to_indices
    
    def __iter__(self):
        """生成平衡的batch"""
        # 为每个难度类别创建循环迭代器
        iterators = {}
        for diff in range(self.num_difficulty_classes):
            if diff in self.difficulty_to_indices and len(self.difficulty_to_indices[diff]) > 0:
                indices = self.difficulty_to_indices[diff].copy()
                np.random.shuffle(indices)  # 随机打乱
                iterators[diff] = self._cycle_iterator(indices)
            else:
                # 如果某个难度类别没有样本，用其他类别代替
                print(f"⚠️ 警告：难度类别{diff}没有样本，将用其他类别代替")
        
        # 生成balanced batches
        all_batch_indices = []
        
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            # 每个难度类别至少包含min_samples_per_class个样本
            for diff in range(self.num_difficulty_classes):
                if diff in iterators:
                    for _ in range(self.min_samples_per_class):
                        if len(batch_indices) < self.batch_size:
                            batch_indices.append(next(iterators[diff]))
            
            # 剩余位置随机填充
            remaining_slots = self.batch_size - len(batch_indices)
            if remaining_slots > 0:
                # 按原始分布比例填充剩余位置
                available_diffs = list(iterators.keys())
                for _ in range(remaining_slots):
                    # 随机选择一个难度类别
                    diff = np.random.choice(available_diffs)
                    batch_indices.append(next(iterators[diff]))
            
            # 随机打乱batch内的顺序
            np.random.shuffle(batch_indices)
            all_batch_indices.extend(batch_indices)
        
        return iter(all_batch_indices)
    
    def _cycle_iterator(self, indices):
        """创建循环迭代器，耗尽后重新开始"""
        while True:
            for idx in indices:
                yield idx
            # 重新打乱顺序
            np.random.shuffle(indices)
    
    def __len__(self):
        """返回总样本数"""
        return self.num_batches * self.batch_size


def create_balanced_dataloader(dataset, batch_size=256, num_workers=0, **kwargs):
    """
    创建使用平衡采样器的DataLoader
    
    Args:
        dataset: CzyKTDataset实例
        batch_size: batch大小
        num_workers: 工作线程数
        **kwargs: 其他DataLoader参数
    
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    
    # 创建平衡采样器
    sampler = DifficultyBalancedSampler(dataset, batch_size)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs
    )
    
    return dataloader 