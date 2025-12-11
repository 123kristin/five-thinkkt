"""
CoT (Chain of Thought) Prompt 模板
用于生成知识推理链的提示词模板
"""
from typing import List, Dict, Optional


def build_cot_prompt(
    history_qids: List[int],
    history_rs: List[int],
    current_qid: int,
    kc_vocab: Dict[int, str],
    history_kcs: Optional[List[List[int]]] = None,
    current_kcs: Optional[List[int]] = None
) -> str:
    """
    构建 CoT 生成提示词
    
    Args:
        history_qids: 历史问题ID列表
        history_rs: 历史答题结果列表 (0/1)
        current_qid: 当前问题ID
        kc_vocab: 知识点词表 {kc_id: kc_name}
        history_kcs: 历史问题的知识点列表（可选）
        current_kcs: 当前问题的知识点列表（可选）
        
    Returns:
        prompt: 完整的提示词字符串
    """
    prompt_parts = []
    
    # 1. 系统提示
    prompt_parts.append("你是一个知识追踪专家，需要分析学生的学习情况并生成推理链。")
    prompt_parts.append("请根据学生的历史答题记录和当前题目，生成结构化的推理链。\n")
    
    # 2. 历史交互信息
    if len(history_qids) > 0:
        prompt_parts.append("## 学生历史交互记录：")
        for i, (qid, r) in enumerate(zip(history_qids[-5:], history_rs[-5:]), 1):  # 只显示最近5条
            result_text = "答对" if r == 1 else "答错"
            kc_text = ""
            if history_kcs and i <= len(history_kcs):
                kcs = history_kcs[-5:][i-1] if len(history_kcs) >= i else []
                kc_names = [kc_vocab.get(kc_id, f"知识点{kc_id}") for kc_id in kcs if kc_id in kc_vocab]
                if kc_names:
                    kc_text = f"，涉及知识点：{', '.join(kc_names)}"
            prompt_parts.append(f"  问题 {qid}: {result_text}{kc_text}")
        prompt_parts.append("")
    
    # 3. 当前题目信息
    prompt_parts.append("## 当前题目：")
    prompt_parts.append(f"问题ID: {current_qid}")
    if current_kcs:
        kc_names = [kc_vocab.get(kc_id, f"知识点{kc_id}") for kc_id in current_kcs if kc_id in kc_vocab]
        if kc_names:
            prompt_parts.append(f"涉及知识点: {', '.join(kc_names)}")
    prompt_parts.append("")
    
    # 4. 推理要求
    prompt_parts.append("## 请生成推理链（按以下结构）：")
    prompt_parts.append("1. **题目考察知识点识别**：当前题目主要考察哪些知识点？")
    prompt_parts.append("2. **学生历史掌握情况**：基于历史记录，学生已掌握/薄弱的知识点有哪些？")
    prompt_parts.append("3. **图像关键信息**：题目图片中包含哪些关键信息（如几何图形、标注、已知量等）？")
    prompt_parts.append("4. **可能错误原因**：如果学生答错，可能的原因是什么？如果答对，说明其掌握情况。")
    prompt_parts.append("5. **预测置信度**：给出0-1之间的置信度分数。")
    prompt_parts.append("")
    prompt_parts.append("**注意**：")
    prompt_parts.append("- 如果图片中没有某个标记或信息，请明确说明'缺失'，不要臆测")
    prompt_parts.append("- 推理链要简洁，控制在80-120个token")
    prompt_parts.append("- 重点关注知识点级别的分析，而非题目级别")
    
    return "\n".join(prompt_parts)


def parse_cot_response(cot_text: str) -> Dict[str, any]:
    """
    解析 CoT 响应，提取结构化信息
    
    Args:
        cot_text: CoT 文本
        
    Returns:
        parsed_info: 包含知识点、置信度等信息的字典
    """
    parsed = {
        'kc_mentioned': [],
        'mastery_mentioned': [],
        'weakness_mentioned': [],
        'confidence': None,
        'error_reason': None
    }
    
    # 简单的关键词提取（可以后续用更复杂的NLP方法）
    cot_lower = cot_text.lower()
    
    # 提取置信度（查找0-1之间的数字）
    import re
    confidence_match = re.search(r'置信度[：:]\s*([0-9.]+)', cot_text)
    if confidence_match:
        try:
            parsed['confidence'] = float(confidence_match.group(1))
        except:
            pass
    
    # 提取"掌握"和"薄弱"的关键词
    if '掌握' in cot_text or '已掌握' in cot_text:
        # 可以进一步用NER提取知识点名称
        pass
    
    return parsed


def validate_cot(cot_text: str, min_length: int = 20, max_length: int = 500) -> bool:
    """
    验证 CoT 文本是否符合要求
    
    Args:
        cot_text: CoT 文本
        min_length: 最小长度
        max_length: 最大长度
        
    Returns:
        is_valid: 是否有效
    """
    if not cot_text or not isinstance(cot_text, str):
        return False
    
    text_len = len(cot_text.strip())
    if text_len < min_length or text_len > max_length:
        return False
    
    # 检查是否包含关键部分
    required_keywords = ['知识点', '掌握', '题目']
    has_keywords = any(keyword in cot_text for keyword in required_keywords)
    
    return has_keywords

