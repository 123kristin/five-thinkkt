"""
Meta-ThinkKT RL Training Script
Phase 3: Train the Meta-Controller using Reinforcement Learning
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts_training2testing/examples'))
from pykt.models import init_model, load_model
from pykt.datasets import init_dataset4train
from pykt.utils import set_seed
from pykt.models.our_model.rl.cot_rl_trainer import CoTRLTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Meta-Controller for ThinkKT")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kt_model_path", type=str, required=True, help="Pre-trained ThinkKT model path")
    parser.add_argument("--lambda_cost", type=float, default=0.1, help="Cost penalty coefficient")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    os.environ['CURRENT_GPU_ID'] = args.gpu_id
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[RL] Using device: {device}")
    
    # 1. Load Pre-trained ThinkKT (Phase 1 Model)
    # Note: We need to load a model that HAS the meta_policy_net structure initialized
    # If loading an old checkpoint without meta_policy, we might need strict=False
    print(f"[RL] Loading model from {args.kt_model_path}...")
    
    # Normally we load config from file
    config_path = os.path.join(args.kt_model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
        
    model_config = config['model_config']
    data_config = config['data_config'] # Adjust based on actual config structure
    
    # Initialize implementation
    # Note: init_model might need adjustment to pass 'use_meta_controller=True' if we made it optional
    # logic assumes ThinkKT class now always has it.
    
    # Mocking data_config for now as we don't have the file structure
    # In real execution, use proper loading
    
    # Use load_model helper
    # We might need to handle 'strict=False' if the checkpoint is from Phase 1 (no policy net weights)
    # The load_model utility in your repo should handle this.
    try:
        model = load_model("thinkkt", model_config, data_config, config['emb_type'], args.kt_model_path)
    except:
        print("[RL] load_model failed, trying initializing and loading state dict...")
        # Fallback logic here if needed
        raise
        
    model.to(device)
    print("[RL] Model loaded successfully.")
    
    # 2. Prepare Data
    train_loader, valid_loader, *_ = init_dataset4train(
        args.dataset_name, "thinkkt", data_config, args.fold, args.batch_size
    )
    
    # 3. Initialize Trainer
    trainer = CoTRLTrainer(
        kt_model=model,
        lambda_cost=args.lambda_cost,
        learning_rate=args.learning_rate
    )
    
    # 4. Training Loop
    print(f"[RL] Starting Training (Lambda={args.lambda_cost})...")
    
    for epoch in range(args.num_epochs):
        metrics_list = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move data to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            # Train Step
            metrics = trainer.train_step(batch)
            metrics_list.append(metrics)
            
            pbar.set_postfix({
                'R': f"{metrics['reward']:.2f}",
                'Rate': f"{metrics['action_rate']:.2f}"
            })
            
        # Epoch Summary
        avg_reward = np.mean([m['reward'] for m in metrics_list])
        avg_rate = np.mean([m['action_rate'] for m in metrics_list])
        print(f"Epoch {epoch}: Avg Reward = {avg_reward:.4f}, Action Rate = {avg_rate:.2%}")
        
    print("[RL] Training Finished.")

if __name__ == "__main__":
    main()
