import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    # accept same arguments as wandb_thinkkt_train.py to avoid errors
    # We only care about params affecting the path
    
    # Needs: dataset_name, model_name, emb_type, save_dir, fold, etc.
    # To be safe, allow unknown args? No, argparse handles it.
    
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str, default="thinkkt")
    parser.add_argument("--emb_type", type=str, default="qkcs")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    
    # ThinkKT specific
    parser.add_argument("--d_question", type=int, default=1024)
    parser.add_argument("--d_cot", type=int, default=384)
    parser.add_argument("--d_knowledge", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_model_type", type=str, default="lstm")
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--mllm_name", type=str, default="")
    parser.add_argument("--use_cot", type=int, default=0)
    parser.add_argument("--use_visual", type=int, default=1)
    parser.add_argument("--question_rep_type", type=str, default="visual")
    parser.add_argument("--cache_dir", type=str, default="features")
    parser.add_argument("--cot_cache_dir", type=str, default="cot_cache")
    parser.add_argument("--cot_threshold", type=int, default=2)
    parser.add_argument("--adaptive_strategy", type=str, default="rule")
    
    # Ignored
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="0")
    
    args, unknown = parser.parse_known_args()
    params = vars(args)
    
    # Logic copied from wandb_train.py
    # params_str construction
    params_for_str = params.copy()
    
    # Check wandb_train.py line 83 exclusions:
    # 'seed', 'save_dir', 'add_uuid', 'other_config', 'emb_path', 'gen_emb_path', 'difficulty_path', 
    # 'gen_kc_emb_file', 'use_wandb', 'num_epochs', 'gpu_id', 'mllm_name', 'cot_cache_dir', 'cache_dir'
    
    excluded_keys = [
        'seed', 'save_dir', 'add_uuid', 'other_config', 'emb_path', 'gen_emb_path', 'difficulty_path',
        'gen_kc_emb_file', 'use_wandb', 'num_epochs', 'gpu_id', 'mllm_name', 'cot_cache_dir', 'cache_dir'
    ]
    
    # In run_bs_experiments, we don't pass 'gen_kc_emb_file', 'other_config' etc. 
    # But params contains default None if not in args? No, argparse defaults.
    
    # Note: wandb_train.py iterates params_for_str.items(). 
    # The ORDER matters. python 3.7+ preserves insertion order.
    # BUT argparse 'vars' order depends on add_argument order?
    # NO. 'vars(args)' treats Namespace as dict.
    # The order of keys in vars(args) is usually insertion order of attributes.
    # Wait, wandb_train.py uses:
    # params_str = "_".join( [str(v) for k,v in params_for_str.items() if k not in excluded] )
    
    # THIS DATA DEPENDENCY IS BRITTLE.
    # If check_completion.py argument order differs from valid run, string differs.
    # However, Python dict order is insertion order.
    # vars(args) returns __dict__.
    
    # To be ROBUST, we should rely on the directory EXISTING.
    # But we don't know the directory name if we can't reproduce the string.
    
    # ALTERNATIVE:
    # Don't try to reproduce path.
    # Just look at `save_dir` (e.g. saved_model/bs/qid).
    # Iterate ALL subdirectories.
    # Load `config.json` from each.
    # Compare `config['params']` with current params?
    # This is 100% robust.
    
    # Implementation 2 (Robust):
    # 1. List all subdirs in args.save_dir.
    # 2. For each subdir:
    #    Read config.json.
    #    Check if params match (carefully).
    #    If match:
    #       Check if .ckpt exists AND predicting.log exists.
    #       If yes -> Return 0 (Found Completed).
    #       If no -> Continue (maybe failed run).
    # 3. If no match found -> Return 1 (Not Found).
    
    target_params = params.copy()
    # Remove transient keys
    for k in ['gpu_id', 'use_wandb', 'add_uuid', 'save_dir']:
        if k in target_params:
            del target_params[k]
        
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        exit(1) # Not found
        
    for dirname in os.listdir(save_dir):
        dirpath = os.path.join(save_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
            
        config_path = os.path.join(dirpath, "config.json")
        if not os.path.exists(config_path):
            continue
            
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                saved_params = saved_config.get('params', {})
                
                # Compare critical keys
                # We care about: model_name, dataset_name, d_question, question_rep_type, num_lstm_layers, fold
                match = True
                keys_to_check = [
                    'model_name', 'dataset_name', 'fold', 'd_question', 
                    'question_rep_type', 'num_lstm_layers', 'use_cot', 'use_visual'
                ]
                for k in keys_to_check:
                    # Cast strings/ints to match types
                    v1 = target_params.get(k)
                    v2 = saved_params.get(k)
                    if str(v1) != str(v2):
                        match = False
                        break
                
                if match:
                    # Found the directory for this config.
                    # Check completion status.
                    ckpt_file = os.path.join(dirpath, f"{args.emb_type}_model.ckpt")
                    if os.path.exists(ckpt_file):
                        # check log starting with predicting
                        has_predict_log = False
                        for fname in os.listdir(dirpath):
                            if fname.startswith("predicting") and fname.endswith(".log"):
                                has_predict_log = True
                                break
                        
                        if has_predict_log:
                            print(f"Skipping: Found completed experiment in {dirname}")
                            exit(0) # Found and Completed
        except Exception as e:
            continue
            
    exit(1) # Not Found / Not Completed

if __name__ == "__main__":
    main()
