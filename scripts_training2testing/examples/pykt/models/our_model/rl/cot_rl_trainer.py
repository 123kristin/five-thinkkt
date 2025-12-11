"""
CoT 强化学习训练器
使用 RL 优化 CoT 生成质量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from ..cot.cot_prompts import parse_cot_response


class CoTRLTrainer:
    """
    CoT 强化学习训练器
    
    使用 GRPO 或 PPO 优化 CoT 生成器的 LoRA 参数
    """
    
    def __init__(
        self,
        cot_generator,
        kt_model,
        reward_weights: Optional[Dict[str, float]] = None,
        use_lora: bool = True
    ):
        """
        初始化 RL 训练器
        
        Args:
            cot_generator: CoT 生成器（需要优化）
            kt_model: 知识追踪模型（冻结，用于计算奖励）
            reward_weights: 奖励权重字典
            use_lora: 是否只优化 LoRA 参数
        """
        self.cot_generator = cot_generator
        self.kt_model = kt_model
        self.use_lora = use_lora
        
        # 冻结 KT 模型
        for param in self.kt_model.parameters():
            param.requires_grad = False
        
        # 奖励权重
        self.reward_weights = reward_weights or {
            'pred': 1.0,  # 预测准确性奖励
            'cons': 0.5,  # 一致性奖励
            'kc': 0.3,    # 知识点覆盖奖励
            'len': 0.1    # 长度惩罚
        }
    
    def compute_reward(
        self,
        cot_texts: List[str],
        predictions_with_cot: torch.Tensor,
        predictions_without_cot: torch.Tensor,
        true_labels: torch.Tensor,
        kc_labels: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        """
        计算奖励
        
        Args:
            cot_texts: CoT 文本列表
            predictions_with_cot: 使用 CoT 的预测结果 (batch, seq_len)
            predictions_without_cot: 不使用 CoT 的预测结果 (batch, seq_len)
            true_labels: 真实标签 (batch, seq_len)
            kc_labels: 知识点标签列表（可选）
            
        Returns:
            rewards: 奖励值 (batch, seq_len)
        """
        batch_size, seq_len = predictions_with_cot.shape
        device = predictions_with_cot.device
        
        rewards = torch.zeros((batch_size, seq_len), device=device)
        
        # 1. 预测准确性奖励 R_pred
        # 计算预测提升（使用 CoT 后的预测更准确）
        pred_improvement = (
            F.binary_cross_entropy(predictions_without_cot, true_labels, reduction='none') -
            F.binary_cross_entropy(predictions_with_cot, true_labels, reduction='none')
        )
        r_pred = pred_improvement * self.reward_weights['pred']
        
        # 2. 一致性奖励 R_cons
        # CoT 中声称的掌握情况应与预测结果一致
        r_cons = self._compute_consistency(cot_texts, predictions_with_cot) * self.reward_weights['cons']
        
        # 3. 知识点覆盖奖励 R_kc
        if kc_labels is not None:
            r_kc = self._compute_kc_coverage(cot_texts, kc_labels) * self.reward_weights['kc']
        else:
            r_kc = torch.zeros((batch_size, seq_len), device=device)
        
        # 4. 长度惩罚 R_len
        r_len = self._compute_length_penalty(cot_texts) * self.reward_weights['len']
        
        # 加权求和
        rewards = r_pred + r_cons + r_kc + r_len
        
        # 归一化到合理范围
        rewards = torch.clamp(rewards, -1.0, 1.0)
        
        return rewards
    
    def _compute_consistency(
        self,
        cot_texts: List[str],
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性奖励
        
        CoT 中声称"掌握"的知识点，预测概率应该较高
        CoT 中声称"薄弱"的知识点，预测概率应该较低
        """
        batch_size, seq_len = predictions.shape
        device = predictions.device
        consistency_scores = torch.zeros((batch_size, seq_len), device=device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                idx = b * seq_len + s
                if idx < len(cot_texts):
                    cot_text = cot_texts[idx]
                    pred_prob = predictions[b, s].item()
                    
                    # 简单的关键词匹配
                    if '掌握' in cot_text or '已掌握' in cot_text:
                        # 如果声称掌握，预测概率应该高
                        consistency_scores[b, s] = pred_prob
                    elif '薄弱' in cot_text or '未掌握' in cot_text:
                        # 如果声称薄弱，预测概率应该低
                        consistency_scores[b, s] = 1.0 - pred_prob
                    else:
                        # 中性情况
                        consistency_scores[b, s] = 0.5
        
        return consistency_scores
    
    def _compute_kc_coverage(
        self,
        cot_texts: List[str],
        kc_labels: List[List[int]]
    ) -> float:
        """
        计算知识点覆盖奖励
        
        CoT 中提及的知识点应与题目标注的知识点重合
        """
        coverage_scores = []
        
        for i, (cot_text, kcs) in enumerate(zip(cot_texts, kc_labels)):
            if not kcs:
                coverage_scores.append(0.0)
                continue
            
            # 简单的关键词匹配（可以后续用NER改进）
            mentioned_count = 0
            for kc_id in kcs:
                # 假设知识点名称包含在 CoT 中
                kc_str = f"知识点{kc_id}"
                if kc_str in cot_text or str(kc_id) in cot_text:
                    mentioned_count += 1
            
            coverage = mentioned_count / len(kcs) if kcs else 0.0
            coverage_scores.append(coverage)
        
        # 转换为 tensor（需要知道 batch 和 seq_len）
        # 这里返回平均值，实际使用时需要reshape
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def _compute_length_penalty(self, cot_texts: List[str]) -> torch.Tensor:
        """
        计算长度惩罚
        
        鼓励 CoT 长度在 80-120 tokens 之间
        """
        penalties = []
        target_length = 100
        min_length = 80
        max_length = 120
        
        for cot_text in cot_texts:
            length = len(cot_text.split())  # 简单的词数统计
            if min_length <= length <= max_length:
                penalty = 0.0  # 理想长度，无惩罚
            elif length < min_length:
                penalty = -0.1 * (min_length - length) / min_length  # 太短，惩罚
            else:
                penalty = -0.1 * (length - max_length) / max_length  # 太长，惩罚
            penalties.append(penalty)
        
        # 转换为 tensor（需要知道 batch 和 seq_len）
        return np.mean(penalties) if penalties else 0.0
    
    def train_step(
        self,
        batch_data: Dict,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        执行一步 RL 训练
        
        Args:
            batch_data: 批次数据
            optimizer: 优化器
            
        Returns:
            metrics: 训练指标字典
        """
        # 这里需要实现 GRPO 或 PPO 的训练逻辑
        # 由于实现较复杂，这里提供框架
        
        # 1. 生成 CoT（带梯度）
        # 2. 计算奖励
        # 3. 计算策略梯度
        # 4. 更新参数
        
        # 简化版本：使用策略梯度
        metrics = {
            'reward': 0.0,
            'loss': 0.0
        }
        
        return metrics

