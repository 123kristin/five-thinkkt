"""
CoT 预生成脚本
批量生成所有交互序列的 CoT 并缓存
"""
import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))
from pykt.models.our_model.cot.cot_generator import CoTGenerator
from pykt.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="预生成 CoT")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_config_path", type=str, required=True, help="数据配置文件路径")
    parser.add_argument("--sequence_file", type=str, required=True, help="序列文件路径")
    parser.add_argument("--mllm_name", type=str,
                       default="/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
                       help="MLLM 模型路径")
    parser.add_argument("--cache_dir", type=str, default="cot_cache", help="缓存目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--max_samples", type=int, default=None, help="最大处理样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CURRENT_GPU_ID'] = args.gpu_id
    
    # 加载数据配置
    import json
    with open(args.data_config_path, 'r') as f:
        data_configs = json.load(f)
        data_config = data_configs[args.dataset_name]
    
    # 加载序列数据
    print(f"[预生成CoT] 正在加载序列数据...")
    df = pd.read_csv(args.sequence_file)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"[预生成CoT] 已加载 {len(df)} 条序列")
    
    # 构建知识点词表（需要从数据中提取）
    kc_vocab = {}  # TODO: 从数据中加载知识点词表
    # 这里需要根据实际数据格式来构建
    
    # 初始化 CoT 生成器
    print(f"[预生成CoT] 正在初始化 CoT 生成器...")
    cot_generator = CoTGenerator(
        mllm_name=args.mllm_name,
        cache_dir=args.cache_dir,
        device=torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    
    # 处理每条序列
    print(f"[预生成CoT] 开始生成 CoT...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成CoT"):
        # 解析序列数据（需要根据实际格式调整）
        # 这里假设数据格式包含 questions, responses 等字段
        try:
            qids = [int(x) for x in str(row.get('questions', '')).split(',')]
            rs = [int(x) for x in str(row.get('responses', '')).split(',')]
            
            # 为序列中的每个位置生成 CoT
            for i in range(len(qids) - 1):
                history_qids = qids[:i+1]
                history_rs = rs[:i+1]
                current_qid = qids[i+1]
                
                # 获取图片路径（需要从数据配置中获取）
                img_path = f"data/{args.dataset_name}/q_imgs/{current_qid}.jpg"
                
                # 生成 CoT
                cot_text, cot_embed = cot_generator.generate_cot(
                    history_qids=history_qids,
                    history_rs=history_rs,
                    current_qid=current_qid,
                    img_path=img_path,
                    kc_vocab=kc_vocab
                )
        except Exception as e:
            print(f"[预生成CoT] 警告: 处理序列 {idx} 失败: {e}")
            continue
    
    # 保存缓存
    print(f"[预生成CoT] 正在保存 CoT 缓存...")
    cot_generator._save_cot_cache()
    
    print(f"[预生成CoT] 完成！")

if __name__ == "__main__":
    main()

