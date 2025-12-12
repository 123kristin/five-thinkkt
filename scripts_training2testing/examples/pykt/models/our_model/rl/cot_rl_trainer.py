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
            r_kc = self._compute_kc_coverage(cot_texts, kc_labels, batch_size, seq_len) * self.reward_weights['kc']
        else:
            r_kc = torch.zeros((batch_size, seq_len), device=device)
        
        # 4. 长度惩罚 R_len
        r_len = self._compute_length_penalty(cot_texts, batch_size, seq_len) * self.reward_weights['len']
        
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
        kc_labels: List[List[int]],
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        计算知识点覆盖奖励
        
        CoT 中提及的知识点应与题目标注的知识点重合
        """
        device = self.kt_model.device if hasattr(self.kt_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coverage_scores = torch.zeros((batch_size, seq_len), device=device)
        
        for i, (cot_text, kcs) in enumerate(zip(cot_texts, kc_labels)):
            if not kcs:
                continue
            
            # 简单的关键词匹配（可以后续用NER改进）
            mentioned_count = 0
            for kc_id in kcs:
                # 假设知识点名称包含在 CoT 中
                kc_str = f"知识点{kc_id}"
                if kc_str in cot_text or str(kc_id) in cot_text:
                    mentioned_count += 1
            
            coverage = mentioned_count / len(kcs) if kcs else 0.0
            b, s = i // seq_len, i % seq_len
            if b < batch_size and s < seq_len:
                coverage_scores[b, s] = coverage
        
        return coverage_scores
    
    def _compute_length_penalty(self, cot_texts: List[str], batch_size: int, seq_len: int) -> torch.Tensor:
        """
        计算长度惩罚
        
        鼓励 CoT 长度在 80-120 tokens 之间
        """
        device = self.kt_model.device if hasattr(self.kt_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        penalties = torch.zeros((batch_size, seq_len), device=device)
        target_length = 100
        min_length = 80
        max_length = 120
        
        for i, cot_text in enumerate(cot_texts):
            length = len(cot_text.split())  # 简单的词数统计
            if min_length <= length <= max_length:
                penalty = 0.0  # 理想长度，无惩罚
            elif length < min_length:
                penalty = -0.1 * (min_length - length) / min_length  # 太短，惩罚
            else:
                penalty = -0.1 * (length - max_length) / max_length  # 太长，惩罚
            
            b, s = i // seq_len, i % seq_len
            if b < batch_size and s < seq_len:
                penalties[b, s] = penalty
        
        return penalties
    
    def train_step(
        self,
        batch_data: Dict,
        optimizer: torch.optim.Optimizer,
        clip_value: float = 0.2,
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """
        执行一步 RL 训练（使用简化的策略梯度方法）
        
        Args:
            batch_data: 批次数据，包含：
                - qseqs: (batch, seq_len) 问题ID序列
                - rseqs: (batch, seq_len) 答题结果序列
                - shft_rseqs: (batch, seq_len) 下一个答题结果（标签）
                - cseqs: (batch, seq_len, max_concepts) 知识点序列（可选）
                - img_path_dict: {qid: img_path} 图片路径映射
                - kc_vocab: {kc_id: kc_name} 知识点词表
            optimizer: 优化器
            clip_value: PPO clip 参数（简化版本中使用）
            gamma: 折扣因子
            
        Returns:
            metrics: 训练指标字典
        """
        # 设置为训练模式
        self.cot_generator.train()
        
        # 获取数据
        qseqs = batch_data['qseqs']  # (batch, seq_len)
        rseqs = batch_data['rseqs']  # (batch, seq_len)
        shft_rseqs = batch_data['shft_rseqs']  # (batch, seq_len) 标签
        cseqs = batch_data.get('cseqs', None)
        img_path_dict = batch_data.get('img_path_dict', {})
        kc_vocab = batch_data.get('kc_vocab', {})
        
        batch_size, seq_len = qseqs.shape
        device = qseqs.device
        
        # 1. 生成 CoT（不使用缓存，使用训练模式）
        cot_texts = []
        cot_embeds_list = []
        
        qseqs_cpu = qseqs.cpu().numpy()
        rseqs_cpu = rseqs.cpu().numpy()
        
        for b in range(batch_size):
            batch_cot_texts = []
            batch_cot_embeds = []
            for s in range(seq_len):
                qid = int(qseqs_cpu[b, s])
                history_qids = [int(qseqs_cpu[b, i]) for i in range(s)]
                history_rs = [int(rseqs_cpu[b, i]) for i in range(s)]
                
                if qid in img_path_dict:
                    img_path = img_path_dict[qid]
                    
                    # 获取知识点
                    history_kcs = None
                    current_kcs = None
                    if cseqs is not None:
                        cseqs_cpu = cseqs.cpu().numpy()
                        history_kcs = [[int(cseqs_cpu[b, i, j]) for j in range(cseqs.shape[2]) 
                                       if cseqs_cpu[b, i, j] >= 0] for i in range(s)]
                        current_kcs = [int(cseqs_cpu[b, s, j]) for j in range(cseqs.shape[2]) 
                                      if cseqs_cpu[b, s, j] >= 0]
                    
                    try:
                        # 生成 CoT（暂时禁用缓存以确保训练）
                        cot_text, cot_embed = self.cot_generator.generate_cot(
                            history_qids=history_qids,
                            history_rs=history_rs,
                            current_qid=qid,
                            img_path=img_path,
                            kc_vocab=kc_vocab,
                            history_kcs=history_kcs,
                            current_kcs=current_kcs
                        )
                        batch_cot_texts.append(cot_text)
                        batch_cot_embeds.append(cot_embed)
                    except Exception as e:
                        print(f"[RL] 警告: 生成CoT失败: {e}")
                        batch_cot_texts.append("")
                        batch_cot_embeds.append(torch.zeros(self.cot_generator.d_cot, device=device))
                else:
                    batch_cot_texts.append("")
                    batch_cot_embeds.append(torch.zeros(self.cot_generator.d_cot, device=device))
            
            cot_texts.extend(batch_cot_texts)
            cot_embeds_list.extend(batch_cot_embeds)
        
        cot_embeds = torch.stack(cot_embeds_list).view(batch_size, seq_len, -1)  # (batch, seq_len, d_cot)
        
        # 2. 使用 KT 模型预测（有CoT和无CoT）
        # 构建 KT 模型的输入
        kt_input = {
            'qseqs': qseqs,
            'rseqs': rseqs,
            'shft_rseqs': shft_rseqs,
            'cseqs': cseqs,
            'smasks': torch.ones_like(qseqs, dtype=torch.bool)
        }
        
        # 计算基础预测（不使用CoT）
        # 临时禁用CoT功能
        original_use_cot = self.kt_model.use_cot
        self.kt_model.use_cot = False
        self.kt_model.eval()
        with torch.no_grad():
            y_no_cot, _ = self.kt_model.train_one_step(kt_input)
        
        # 使用CoT的预测（临时使用我们生成的CoT）
        # 注意：这里需要修改KT模型以支持动态传入CoT嵌入
        # 简化版本：使用生成的CoT嵌入通过KT网络直接预测
        self.kt_model.use_cot = True
        # 由于ThinkKT的train_one_step会调用_get_cot_embeddings，我们需要临时替换cot_generator
        # 或者直接修改KT网络的输入
        
        # 简化处理：直接使用KT网络的forward，手动传入CoT嵌入
        # 获取题目特征和知识点分布
        v_t, k_t = self.kt_model._get_question_features(qseqs, seq_len)
        
        # 使用生成的CoT嵌入
        r_embed = cot_embeds  # (batch, seq_len, d_cot)
        
        # KT网络前向传播
        self.kt_model.kt_net.train()
        y_with_cot = self.kt_model.kt_net(
            v_t=v_t,
            a_t=rseqs,
            k_t=k_t,
            r_embed=r_embed,
            mask=None
        )
        
        # 恢复原始状态
        self.kt_model.use_cot = original_use_cot
        
        # 计算奖励
        rewards = self.compute_reward(
            cot_texts=cot_texts,
            predictions_with_cot=y_with_cot,
            predictions_without_cot=y_no_cot,
            true_labels=shft_rseqs,
            kc_labels=[cseqs_cpu[b, s].tolist() if cseqs is not None else [] 
                      for b in range(batch_size) for s in range(seq_len)]
        )
        
        # 4. 计算策略梯度损失（REINFORCE 简化版本）
        # 注意：这是一个简化实现。由于 MLLM 生成是离散的，完整的RL训练需要：
        # 1. 使用 LoRA 微调 MLLM，并通过 logits 计算 log-probability
        # 2. 或使用 GRPO/PPO 等序列级RL算法
        # 当前版本使用 CoT 嵌入作为代理，主要优化文本编码器参数
        
        # 计算 CoT 嵌入的 log_prob（简化：使用嵌入的L2范数作为代理）
        # 实际应该使用 MLLM 生成时的 logits 计算 log_probs
        log_probs = -0.5 * torch.sum(cot_embeds ** 2, dim=-1)  # (batch, seq_len)
        
        # REINFORCE 损失
        advantages = rewards - rewards.mean()  # 基线减除
        loss = -(log_probs * advantages).mean()
        
        # 5. 反向传播和更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cot_generator.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 计算指标
        metrics = {
            'reward': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'loss': loss.item(),
            'advantage_mean': advantages.mean().item()
        }
        
        return metrics

