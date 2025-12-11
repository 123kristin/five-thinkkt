# CzyKT模型的生成配置文件
# 包含用于训练和推理的默认配置参数

GENERATION_CONFIG = {
    # 数据集配置
    "dataset_name": "DBE_KT22",
    "dataset_path": "data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv",
    
    # 嵌入配置
    "use_content_emb": True,
    "use_analysis_emb": True,
    "use_contrastive": True,
    "contrastive_weight": 0.1,
    "embeddings_base_path": "data",
    "content_emb_filename": "content.pt",
    "analysis_emb_filename": "analysis.pt",
    
    # 模型配置
    "emb_size": 200,
    "dropout": 0.2,
    "pretrain_dim": 1536,
    "content_dim": 512,
    "analysis_dim": 1536,
    
    # 训练配置
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "early_stopping": 10,
    
    # 其他配置
    "cuda_device": "cuda:0",
    "seed": 42,
    "use_wandb": False,
    "save_model": True,
    
    # GPU配置
    "cuda_visible_devices": "0",
    
    # 实验配置
    "max_new_tokens": 16,
    "is_static": True,
} 