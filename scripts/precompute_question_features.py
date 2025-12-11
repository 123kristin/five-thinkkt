"""
题目特征预计算脚本
批量提取所有题目的视觉特征并缓存
"""
import os
import sys
import argparse
import torch
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))
from pykt.models.our_model.visual_language_encoder import VisualLanguageEncoder, build_img_path_dict
from pykt.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="预计算题目特征")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_config_path", type=str, required=True, help="数据配置文件路径")
    parser.add_argument("--model_path", type=str, 
                       default="/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct",
                       help="视觉模型路径")
    parser.add_argument("--cache_dir", type=str, default="features", help="缓存目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CURRENT_GPU_ID'] = args.gpu_id
    
    # 加载数据配置
    import json
    with open(args.data_config_path, 'r') as f:
        data_configs = json.load(f)
        data_config = data_configs[args.dataset_name]
    
    # 构建图片路径映射
    print(f"[预计算] 正在构建图片路径映射...")
    img_path_dict = build_img_path_dict(args.dataset_name, data_config)
    print(f"[预计算] 已找到 {len(img_path_dict)} 个题目图片")
    
    # 初始化编码器
    print(f"[预计算] 正在初始化视觉编码器...")
    encoder = VisualLanguageEncoder(
        num_c=data_config.get('num_c', 100),
        d_question=1024,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        use_cache=True,
        device=torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    
    # 批量处理
    print(f"[预计算] 开始提取特征...")
    qids = sorted(img_path_dict.keys())
    img_paths = [img_path_dict[qid] for qid in qids]
    
    # 分批处理
    for i in tqdm(range(0, len(qids), args.batch_size), desc="提取特征"):
        batch_qids = qids[i:i+args.batch_size]
        batch_paths = img_paths[i:i+args.batch_size]
        
        # 提取特征（会自动缓存）
        encoder.encode_batch(batch_paths, batch_qids)
    
    # 保存缓存
    print(f"[预计算] 正在保存特征缓存...")
    encoder.save_feature_cache()
    
    print(f"[预计算] 完成！已处理 {len(qids)} 个题目")

if __name__ == "__main__":
    main()

