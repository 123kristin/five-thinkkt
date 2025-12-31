import os
import sys
import argparse

# 先解析参数，设置环境变量
parser = argparse.ArgumentParser()
# dataset config
parser.add_argument("--dataset_name", type=str, default="DBE_KT22")
parser.add_argument("--fold", type=int, default=0)

# train config
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=200)

# log config & save config
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--add_uuid", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="saved_model")

# model config
parser.add_argument("--model_name", type=str, default="vcrkt")
parser.add_argument("--emb_type", type=str, default='qkcs')

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--dim_qc", type=int, default=200, help="dimension of q and c embedding")
parser.add_argument("--question_rep_type", type=str, default='qid', help="question representation type: qid, visual, vq, gf, ca, cl, or cga")
parser.add_argument("--cl_weight", type=float, default=0.1, help="weight for contrastive loss in cl mode")
parser.add_argument("--d_question", type=int, default=1024, help="dimension of visual features")
parser.add_argument("--seq_len", type=int, default=None, help="Sequence length (defaults to config if None)")

# LoRA Arguments
parser.add_argument("--use_lora", type=int, default=0, help="whether to use QLoRA fine-tuning (1=True, 0=False)")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--mllm_name", type=str, default='/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct', help="Path to Qwen2-VL model")

# GPU选择参数 - 新增
parser.add_argument("--gpu_id", type=str, default="0",
                   help="指定使用的GPU ID，如'0','1','2'等")

args = parser.parse_args()

# 设置环境变量 - 只设置CURRENT_GPU_ID，不设置CUDA_VISIBLE_DEVICES
os.environ['CURRENT_GPU_ID'] = args.gpu_id

# 现在才可以导入 torch 及其依赖
import wandb_train
# 修正配置路径，使其适配当前环境
wandb_train.dir_path_of_configs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../my_configs"))
from wandb_train import main
from utils4running import Tee

if __name__ == "__main__":
    params = vars(args)
    
    print(f"实验配置:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  模型名称: {args.model_name}")
    print(f"  使用GPU: cuda:{args.gpu_id}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  训练轮数: {args.num_epochs}")
    
    with Tee(f"{args.save_dir}/training_log/{args.model_name}_training.log"):
        main(params)
