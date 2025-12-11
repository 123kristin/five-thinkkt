import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="XES3G5M")
parser.add_argument("--model_name", type=str, default="qdkt")
parser.add_argument("--emb_type", type=str, default="qid")
parser.add_argument("--save_dir", type=str, default="saved_model")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--emb_size", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=200)

parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--add_uuid", type=int, default=0)

# GPU选择参数
parser.add_argument("--gpu_id", type=str, default="2",
                help="指定使用的GPU ID，如'0','1','2'等")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 必须在import torch前设置
os.environ['CURRENT_GPU_ID'] = args.gpu_id        # 可选，兼容旧逻辑
args = parser.parse_args()

# 现在才可以导入 torch 及其依赖
from wandb_train import main
from utils4running import Tee
from configs.generation_config import GENERATION_CONFIG
if __name__ == "__main__":
    params = vars(args)
    main(params)
    with Tee(f"{args.save_dir}/training_log/{args.model_name}_training.log"):
        main(params)
