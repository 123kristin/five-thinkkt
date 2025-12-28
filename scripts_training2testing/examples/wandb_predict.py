import os
import sys
import argparse

# 先解析参数，设置环境变量
parser = argparse.ArgumentParser()
parser.add_argument("--bz", type=int, default=128)
parser.add_argument("--save_dir", type=str, default="saved_model/baseline_version/nips_task34_0_0.0001_32_thinkkt_qkcs_1024_384_512_0.1_transformer_2_8_2_False_True_features")
parser.add_argument("--fusion_type", type=str, default="late_fusion")
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--gpu_id", type=str, default="0", help="指定使用的GPU ID，如'0','1','2'等")
parser.add_argument("--d_question", type=int, default=1024)
parser.add_argument("--dim_qc", type=int, default=200)
parser.add_argument("--question_rep_type", type=str, default="qid")
parser.add_argument("--cl_weight", type=float, default=0.1)
args = parser.parse_args()

# 设置GPU环境变量，必须在import torch前
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
os.environ['CURRENT_GPU_ID'] = args.gpu_id
print(f"设置GPU环境变量: CUDA_VISIBLE_DEVICES={args.gpu_id}")

import json
import copy
import torch
import pandas as pd

from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets
from utils4running import Tee

dir_path_of_configs = "/home3/zhiyu/code-4/kt_analysis_generation/my_configs"
wandb_config_path = os.path.join(dir_path_of_configs, "wandb.json")

with open(wandb_config_path) as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] == 1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")
    print(batch_size)

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    data_config_path = os.path.join(dir_path_of_configs, "data_config.json")
    with open(data_config_path) as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
        
            # 为DKT模型添加嵌入路径配置
    if model_name == "dkt":
        # DKT模型会根据dataset_name自动推断gen_emb_path，所以不需要显式传递
        # 但是需要确保所有必要的参数都正确传递
        if "dataset_name" in trained_params:
            model_config["dataset_name"] = trained_params["dataset_name"]
        if "content_type" in trained_params:
            model_config["content_type"] = trained_params["content_type"]
        if "analysis_type" in trained_params:
            model_config["analysis_type"] = trained_params["analysis_type"]
        if "cross_attention_layers" in trained_params:
            model_config["cross_attention_layers"] = trained_params["cross_attention_layers"]
        if "attention_type" in trained_params:
            model_config["attention_type"] = trained_params["attention_type"]
        if "no_analysis_fusion" in trained_params:
            model_config["no_analysis_fusion"] = trained_params["no_analysis_fusion"]
        if "use_content_emb" in trained_params:
            model_config["use_content_emb"] = trained_params["use_content_emb"]
        if "use_analysis_emb" in trained_params:
            model_config["use_analysis_emb"] = trained_params["use_analysis_emb"]
        if "use_kc_emb" in trained_params:
            model_config["use_kc_emb"] = trained_params["use_kc_emb"]
        if "trainable_content_emb" in trained_params:
            model_config["trainable_content_emb"] = trained_params["trainable_content_emb"]
        if "trainable_analysis_emb" in trained_params:
            model_config["trainable_analysis_emb"] = trained_params["trainable_analysis_emb"]
        if "trainable_kc_emb" in trained_params:
            model_config["trainable_kc_emb"] = trained_params["trainable_kc_emb"]
        if "content_dim" in trained_params:
            model_config["content_dim"] = trained_params["content_dim"]
        if "analysis_dim" in trained_params:
            model_config["analysis_dim"] = trained_params["analysis_dim"]
        
        print(f"[DKT] 配置参数:")
        print(f"  - dataset_name: {model_config.get('dataset_name', 'None')}")
        print(f"  - content_type: {model_config.get('content_type', 'None')}")
        print(f"  - analysis_type: {model_config.get('analysis_type', 'None')}")
        print(f"  - attention_type: {model_config.get('attention_type', 'None')}")
        print(f"  - cross_attention_layers: {model_config.get('cross_attention_layers', 'None')}")
        print(f"  - no_analysis_fusion: {model_config.get('no_analysis_fusion', 'None')}")

    if model_name not in ["dimkt"]:
        # 创建args对象来传递difficulty_path等参数
        class Args:
            def __init__(self, params):
                for key, value in params.items():
                    setattr(self, key, value)
        # 从训练时的参数中提取difficulty_path
        args_obj = Args(trained_params)
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, args=args_obj)
    else:
        diff_level = trained_params["difficult_levels"]
        args_obj = Args(trained_params)  # 为一致性也创建args对象
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level, args=args_obj)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, model.emb_type + "_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type + "_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }

    q_testaucs, q_testaccs = -1, -1
    qw_testaucs, qw_testaccs = -1, -1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type + "_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc" + key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc" + key] = q_testaccs[key]

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type + "_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc" + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc" + key] = qw_testaccs[key]

    print(dres)
    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config['params'])

    if params['use_wandb'] == 1:
        import wandb
        wandb.log(dres)
    
    # 添加预测完成标记
    print("预测完成")

if __name__ == "__main__":
    # 这里已提前解析参数和设置环境变量，无需重复
    params = vars(args)
    with Tee(f"{args.save_dir}/predicting.log"):
        main(params)