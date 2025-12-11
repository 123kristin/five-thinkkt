import os
import sys
import argparse

# 先解析参数，设置环境变量
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="XES3G5M", 
                   choices=["XES3G5M", "DBE_KT22"],
                   help="选择数据集: XES3G5M 或 DBE_KT22")
parser.add_argument("--model_name", type=str, default="dkt")
parser.add_argument("--emb_type", type=str, default="qid")
parser.add_argument("--save_dir", type=str, default="saved_model")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--emb_size", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=1e-3)

parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--add_uuid", type=int, default=0)

# 实验类型参数 - 新增
parser.add_argument("--experiment_type", type=str, default="basic", 
                   choices=["basic", "enhanced_fixed", "enhanced_trainable", "enhanced_no_analysis", "enhanced_trainable_no_analysis"],
                   help="实验类型: basic=基本DKT, enhanced_fixed=增强DKT(嵌入固定), enhanced_trainable=增强DKT(嵌入可训练), enhanced_no_analysis=增强DKT(不拼接解析嵌入), enhanced_trainable_no_analysis=增强DKT(嵌入可训练且不拼接解析嵌入)")

# 嵌入相关参数 - 简化
parser.add_argument("--content_type", type=str, default="text", 
                   choices=["text", "image"],
                   help="选择内容嵌入类型: text使用文本内容嵌入, image使用图像内容嵌入 (仅enhanced实验需要)")
parser.add_argument("--analysis_type", type=str, default="generated", 
                   choices=["generated", "original"],
                   help="选择解析嵌入类型: generated使用生成解析, original使用原始解析")

# 交叉注意力参数 - 新增
parser.add_argument("--cross_attention_layers", type=int, default=1,
                   help="交叉注意力层数: 1=单层, 2=双层, 3=三层等 (仅enhanced实验需要)")

# 注意力类型参数 - 新增
parser.add_argument("--attention_type", type=str, default="cross", 
                   choices=["cross", "improved_cross", "hierarchical", "knowledge_aware", "three_layer_hierarchical"],
                   help="注意力类型: cross=原始交叉注意力, improved_cross=改进交叉注意力, hierarchical=分层注意力, knowledge_aware=知识感知注意力, three_layer_hierarchical=三层分层注意力")

# GPU选择参数
parser.add_argument("--gpu_id", type=str, default="1",
                   help="指定使用的GPU ID，如'0','1','2'等")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 必须在import torch前设置
os.environ['CURRENT_GPU_ID'] = args.gpu_id        # 可选，兼容旧逻辑

# 现在才可以导入 torch 及其依赖
from wandb_train import main
from utils4running import Tee

"""
DKT模型训练数学公式

1. 模型前向传播:
   y_t = sigmoid(W_out * LSTM(h_{t-1}, x_t))
   其中 x_t = [xemb_t; pretrain_emb_t] 或 x_t = xemb_t

2. 损失函数 (二元交叉熵):
   L = -Σ_{t=1}^T Σ_{c=1}^C [r_{t,c} * log(y_{t,c}) + (1-r_{t,c}) * log(1-y_{t,c})]
   其中:
   - r_{t,c}: 时间步t概念c的真实掌握情况 (0或1)
   - y_{t,c}: 时间步t概念c的预测掌握概率 [0,1]
   - T: 序列长度
   - C: 概念数量

3. 优化目标:
   min_θ L(θ) + λ||θ||_2^2
   其中:
   - θ: 模型参数
   - λ: L2正则化系数
   - ||θ||_2^2: L2正则化项

4. 评估指标:
   - AUC: Area Under the ROC Curve
   - ACC: Accuracy = (TP + TN) / (TP + TN + FP + FN)
   其中 TP, TN, FP, FN 分别为真阳性、真阴性、假阳性、假阴性
"""

def get_embedding_dimensions(dataset_name, content_type):
    """根据数据集和内容类型获取嵌入维度"""
    if dataset_name == "XES3G5M":
        if content_type == "text":
            return 1536  # XES3G5M文本内容嵌入维度
        elif content_type == "image":
            return 512   # XES3G5M图像内容嵌入维度
    elif dataset_name == "DBE_KT22":
        return 1536  # DBE_KT22使用text-embedding-3-small模型，维度为1536
    
    # 默认值
    return 512

# 根据实验类型自动设置嵌入参数
def set_experiment_params(args):
    """根据实验类型自动设置嵌入参数"""
    if args.experiment_type == "basic":
        # 实验1: 基本DKT模型
        args.use_content_emb = 0
        args.use_analysis_emb = 0
        args.use_kc_emb = 0
        args.trainable_content_emb = 0
        args.trainable_analysis_emb = 0
        args.trainable_kc_emb = 0
        args.content_dim = 512  # 基本实验不使用，设为默认值
        args.analysis_dim = 1536
        # basic实验不需要content_type，设为默认值
        args.content_type = "text"
        # basic实验不使用交叉注意力
        args.cross_attention_layers = 1
        # basic实验使用默认交叉注意力，但保留命令行传入的值
        if not hasattr(args, 'attention_type') or args.attention_type is None:
            args.attention_type = "cross"
        
    elif args.experiment_type == "enhanced_fixed":
        # 实验2: 增强DKT模型，嵌入固定
        args.use_content_emb = 1
        args.use_analysis_emb = 1
        args.use_kc_emb = 1
        args.trainable_content_emb = 0
        args.trainable_analysis_emb = 0
        args.trainable_kc_emb = 0
        # 根据数据集和内容类型自动设置维度
        args.content_dim = get_embedding_dimensions(args.dataset_name, args.content_type)
        args.analysis_dim = 1536
        # 使用指定的交叉注意力层数
        if not hasattr(args, 'cross_attention_layers') or args.cross_attention_layers < 1:
            args.cross_attention_layers = 1
        # 保留命令行传入的注意力类型，不强制覆盖
        
    elif args.experiment_type == "enhanced_trainable":
        # 实验3: 增强DKT模型，嵌入可训练
        args.use_content_emb = 1
        args.use_analysis_emb = 1
        args.use_kc_emb = 1
        args.trainable_content_emb = 1
        args.trainable_analysis_emb = 1
        args.trainable_kc_emb = 1
        # 根据数据集和内容类型自动设置维度
        args.content_dim = get_embedding_dimensions(args.dataset_name, args.content_type)
        args.analysis_dim = 1536
        # 使用指定的交叉注意力层数
        if not hasattr(args, 'cross_attention_layers') or args.cross_attention_layers < 1:
            args.cross_attention_layers = 1
        # 保留命令行传入的注意力类型，不强制覆盖
    
    elif args.experiment_type == "enhanced_no_analysis":
        # 新增: 交叉注意力后不拼接解析嵌入
        args.use_content_emb = 1
        args.use_analysis_emb = 1
        args.use_kc_emb = 1
        args.trainable_content_emb = 0
        args.trainable_analysis_emb = 0
        args.trainable_kc_emb = 0
        args.content_dim = get_embedding_dimensions(args.dataset_name, args.content_type)
        args.analysis_dim = 1536
        args.no_analysis_fusion = 1
        if not hasattr(args, 'cross_attention_layers') or args.cross_attention_layers < 1:
            args.cross_attention_layers = 1
        # 保留命令行传入的注意力类型，不强制覆盖
    elif args.experiment_type == "enhanced_trainable_no_analysis":
        # 新增: 嵌入可训练且不拼接解析嵌入
        args.use_content_emb = 1
        args.use_analysis_emb = 1
        args.use_kc_emb = 1
        args.trainable_content_emb = 1
        args.trainable_analysis_emb = 1
        args.trainable_kc_emb = 1
        args.content_dim = get_embedding_dimensions(args.dataset_name, args.content_type)
        args.analysis_dim = 1536
        args.no_analysis_fusion = 1
        if not hasattr(args, 'cross_attention_layers') or args.cross_attention_layers < 1:
            args.cross_attention_layers = 1
        # 保留命令行传入的注意力类型，不强制覆盖
    return args

# 设置实验参数
args = set_experiment_params(args)

# 将dataset_name添加到参数中，以便传递给模型
if not hasattr(args, 'dataset_name'):
    args.dataset_name = args.dataset_name

if __name__ == "__main__":
    params = vars(args)
    print(f"实验配置:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  实验类型: {args.experiment_type}")
    if args.experiment_type != "basic":
        print(f"  内容类型: {args.content_type}")
        print(f"  内容嵌入维度: {args.content_dim}")
        print(f"  注意力类型: {args.attention_type}")
        print(f"  注意力层数: {args.cross_attention_layers}")
    print(f"  使用内容嵌入: {args.use_content_emb}")
    print(f"  使用解析嵌入: {args.use_analysis_emb}")
    print(f"  使用KC嵌入: {args.use_kc_emb}")
    print(f"  内容嵌入可训练: {args.trainable_content_emb}")
    print(f"  解析嵌入可训练: {args.trainable_analysis_emb}")
    print(f"  KC嵌入可训练: {args.trainable_kc_emb}")
    
    with Tee(f"{args.save_dir}/training_log/{args.model_name}_training.log"):
        main(params)