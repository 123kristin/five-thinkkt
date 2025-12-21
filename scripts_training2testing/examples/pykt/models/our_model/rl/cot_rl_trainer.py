"""
CoT RL Trainer for Meta-Controller
用于训练 Meta-Controller (System 2 Gateway)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np

class CoTRLTrainer:
    """
    Meta-Controller RL 训练器
    使用 REINFORCE 算法优化是否生成 CoT 的策略
    """
    
    def __init__(
        self,
        kt_model,
        cot_generator=None,
        lambda_cost: float = 0.1,
        learning_rate: float = 1e-4
    ):
        """
        Args:
            kt_model: ThinkKT 模型 (包含 Meta-Controller)
            cot_generator: CoT 生成器 (可选，如果需要实时生成)
            lambda_cost: 计算成本系数 (惩罚因子)
        """
        self.model = kt_model
        # 我们只优化 Meta-Controller
        self.optimizer = torch.optim.Adam(self.model.meta_policy_net.parameters(), lr=learning_rate)
        
        self.lambda_cost = lambda_cost
        
    def train_step(self, batch_data: Dict) -> Dict[str, float]:
        """
        执行一步 RL 训练
        """
        self.model.train()
        
        # 1. 前向传播 (Sampling Actions)
        # predictions: (batch, seq_len, 1)
        # actions: (batch, seq_len, 1) -> 0 or 1
        # log_probs: (batch, seq_len, 1)
        predictions, actions, log_probs = self.model.forward_rl(batch_data)
        
        # 移除最后一维
        predictions = predictions.squeeze(-1)
        actions = actions.squeeze(-1)
        log_probs = log_probs.squeeze(-1)
        
        target = batch_data['shft_rseqs']
        mask = batch_data.get('smasks', torch.ones_like(target, dtype=torch.bool))
        
        # 2. 计算 Reward (For each step t)
        # R_acc: 预测准确奖励
        # 简单奖励: 如果预测概率 > 0.5 且 正确 -> +1, 否则 -1
        # 或者使用负 Cross Entropy 作为软奖励 (更加平滑)
        
        # 为了稳定，我们使用 binary cross entropy 的负数
        # R = - BCELoss(pred, target)
        bce_loss = F.binary_cross_entropy(predictions, target.float(), reduction='none')
        r_acc = -bce_loss # Loss越小，Reward越大 (如 -0.1 > -0.5)
        
        # C_cost: 成本惩罚
        # Cost = Action * lambda
        c_cost = actions.float() * self.lambda_cost
        
        # Total Reward
        rewards = r_acc - c_cost
        
        # 3. Policy Gradient Loss
        # Loss = - log_prob * Reward
        # 只计算有效位置 (Mask)
        pg_loss = -(log_probs * rewards)
        pg_loss = (pg_loss * mask).sum() / mask.sum()
        
        # 4. Update
        self.optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.meta_policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': pg_loss.item(),
            'reward': rewards[mask].mean().item(),
            'acc_reward': r_acc[mask].mean().item(),
            'cost_penalty': c_cost[mask].mean().item(),
            'action_rate': actions[mask].float().mean().item() # 平均生成率
        }
