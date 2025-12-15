import os
import sys
import argparse

# 先解析参数，设置环境变量
parser = argparse.ArgumentParser(description="ThinkKT 模型训练入口")
# dataset config
parser.add_argument("--dataset_name", type=str, default="XES3G5M",
                   help="数据集名称: DBE_KT22 或 XES3G5M")
parser.add_argument("--fold", type=int, default=0,
                   help="交叉验证折数 (0-4)")

# train config
parser.add_argument("--learning_rate", type=float, default=1e-4,
                   help="学习率")
parser.add_argument("--seed", type=int, default=2025,
                   help="随机种子")
parser.add_argument("--batch_size", type=int, default=32,
                   help="批次大小（ThinkKT使用视觉模型，建议较小batch size）")
parser.add_argument("--num_epochs", type=int, default=200,
                   help="训练轮数")

# log config & save config
parser.add_argument("--use_wandb", type=int, default=0,
                   help="是否使用wandb记录 (0/1)")
parser.add_argument("--add_uuid", type=int, default=1,
                   help="是否添加UUID到保存路径 (0/1)")
parser.add_argument("--save_dir", type=str, default="saved_model/base",
                   help="模型保存目录")

# model config
parser.add_argument("--model_name", type=str, default="thinkkt",
                   help="模型名称（固定为thinkkt）")
parser.add_argument("--emb_type", type=str, default='qkcs',
                   help="嵌入类型（兼容pykt接口）")

# ThinkKT 特定参数
parser.add_argument("--d_question", type=int, default=1024,
                   help="题目特征维度")
parser.add_argument("--d_cot", type=int, default=384,
                   help="CoT嵌入维度（当前未使用）")
parser.add_argument("--d_knowledge", type=int, default=512,
                   help="知识状态维度")
parser.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout率")
parser.add_argument("--seq_model_type", type=str, default="transformer",
                   choices=["transformer", "lstm"],
                   help="序列模型类型: transformer 或 lstm")
parser.add_argument("--num_transformer_layers", type=int, default=2,
                   help="Transformer层数")
parser.add_argument("--num_heads", type=int, default=8,
                   help="注意力头数")
parser.add_argument("--num_lstm_layers", type=int, default=2,
                   help="LSTM层数（当seq_model_type=lstm时使用）")
parser.add_argument("--mllm_name", type=str, default="/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/Qwen/Qwen2-VL-3B-Instruct",
                   help="多模态大语言模型路径（本地路径或HuggingFace模型名）")
parser.add_argument("--use_cot", type=int, default=0,
                   help="是否使用CoT (0/1，当前版本为0)")
parser.add_argument("--use_visual", type=int, default=1,
                   help="是否使用视觉特征 (0/1)")
parser.add_argument("--cache_dir", type=str, default="features",
                   help="特征缓存目录")

# GPU选择参数
parser.add_argument("--gpu_id", type=str, default="0",
                   help="指定使用的GPU ID，如'0','1','2'等")

args = parser.parse_args()

# 设置环境变量 - 只设置CURRENT_GPU_ID，不设置CUDA_VISIBLE_DEVICES
os.environ['CURRENT_GPU_ID'] = args.gpu_id

# 现在才可以导入 torch 及其依赖
from wandb_train import main
from utils4running import Tee

if __name__ == "__main__":
    params = vars(args)
    
    # 确保模型名称为thinkkt
    params['model_name'] = 'thinkkt'
    
    # 转换use_cot和use_visual为布尔值（wandb_train中可能需要）
    params['use_cot'] = bool(params['use_cot'])
    params['use_visual'] = bool(params['use_visual'])
    
    print("=" * 60)
    print("ThinkKT 模型训练配置")
    print("=" * 60)
    print(f"  数据集: {args.dataset_name}")
    print(f"  模型名称: {params['model_name']}")
    print(f"  使用GPU: cuda:{args.gpu_id}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  题目特征维度: {args.d_question}")
    print(f"  知识状态维度: {args.d_knowledge}")
    print(f"  序列模型类型: {args.seq_model_type}")
    print(f"  使用视觉特征: {args.use_visual}")
    print(f"  使用CoT: {args.use_cot}")
    print(f"  视觉模型: {args.mllm_name}")
    print(f"  特征缓存目录: {args.cache_dir}")
    print("=" * 60)
    
    # 创建日志目录
    log_dir = f"{args.save_dir}/training_log"
    os.makedirs(log_dir, exist_ok=True)
    
    with Tee(f"{log_dir}/thinkkt_training.log"):
        main(params)

