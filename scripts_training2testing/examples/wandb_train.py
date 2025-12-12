import os
import json
import copy
from datetime import datetime
# 其余与torch无关的import保留在顶部
from pykt.utils import debug_print,set_seed

dir_path_of_configs = "/home3/zhiyu/code-4/kt_analysis_generation/my_configs"

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def main(params):
    global device
    if "use_wandb" not in params:
        params['use_wandb'] = 1
    if params['use_wandb']==1:
        import wandb
        wandb.init()
    # 设置GPU设备
    gpu_id = params.get("gpu_id", "1")
    os.environ['CURRENT_GPU_ID'] = gpu_id
    # 这里才import torch及其依赖
    import torch
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"torch.cuda.get_device_name({i}):", torch.cuda.get_device_name(i))
    torch.set_num_threads(4)
    from torch.optim import SGD, Adam
    from pykt.models import train_model,init_model
    from pykt.datasets import init_dataset4train
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Set CURRENT_GPU_ID environment variable to: {gpu_id}")
    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
    debug_print(text = "load config files.",fuc_name="main")
    kt_config_path = os.path.join(dir_path_of_configs, "kt_config.json")
    with open(kt_config_path) as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt","stablekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM
        if model_name in ["thinkkt"]:
            train_config["batch_size"] = 32 ## because of OOM (visual model needs more memory)
        model_config = copy.deepcopy(params)
        for key in ["model_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        # 确保dataset_name在model_config中，以便传递给模型（ThinkKT需要）
        if 'dataset_name' not in model_config:
            model_config['dataset_name'] = dataset_name
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]
    data_config_path = os.path.join(dir_path_of_configs, "data_config.json")
    with open(data_config_path) as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]
    params_for_str = params.copy()
    if 'gen_kc_emb_file' in params_for_str:
        if params_for_str['gen_kc_emb_file']:
            params_for_str['gen_kc_emb_file'] = os.path.splitext(os.path.basename(params_for_str['gen_kc_emb_file']))[0]
        else:
            params_for_str['gen_kc_emb_file'] = 'none'
    params_str = "_".join( [str(v) for k,v in params_for_str.items() if k not in ['seed', 'save_dir', 'add_uuid', 'other_config', 'emb_path', 'gen_emb_path', 'difficulty_path', 'gen_kc_emb_file', 'use_wandb', 'num_epochs', 'gpu_id', 'mllm_name']] )
    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params['use_wandb'] == 1:
        params_str = params_str+f"_{ datetime.now().strftime('%Y%m%d_%H%M%S_%f') }"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    debug_print(text="init_dataset",fuc_name="main")
    class Args:
        def __init__(self, params):
            for key, value in params.items():
                setattr(self, key, value)
    args = Args(params)
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, args=args)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level, args=args)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")
    if model_name in ["dimkt"]:
        del model_config['weight_decay']
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2','gpu_id']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt", "bakt_time","folibikt"]:
        model_config["seq_len"] = seq_len
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print(f"dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    debug_print(text = "train model",fuc_name="main")
    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model,
            train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, 
            data_config[dataset_name], fold)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, 
            train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
        best_model.load_state_dict(net)
    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    print(f"end:{datetime.now()}")
    if params['use_wandb']==1:
        wandb.log({ 
                    "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
