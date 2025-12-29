#!/usr/bin/env python3
import sys
from contextlib import redirect_stdout, redirect_stderr
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import json


# 定义一个上下文管理器，用于同时将输出写入文件和终端
class Tee:
    def __init__(self, filename):
        # 先获取目录，若目录不存在则创建
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # add a timestamp to the params_str, 年月日时分秒微秒
        params_str = f"{ datetime.now().strftime('%Y%m%d_%H%M%S_%f') }"
        filename = filename.replace(".log", f"_{params_str}.log")
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 退出时写入一行横线
        self.file.write("\n" + "-" * 80 + "\n")
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 使用上下文管理器
# with Tee("output.log"):
#     print("This is a test message.")
#     print("This is another line of output.")
#     print("This is an error message.", file=sys.stderr)

def get_id_ori2new(data_config: dict):
    dpath = data_config["dpath"]
    file_keyid2idx = os.path.join(dpath, "keyid2idx.json")
    if not os.path.exists(file_keyid2idx): # 有些数据集不用重映射，比如XES3G5M
        return None, None

    with open(file_keyid2idx, "r") as f:
        keyid2idx = json.load(f)
    qid_ori2new = keyid2idx["questions"]
    assert len(qid_ori2new) == data_config[
        "num_q"], f"len(qid_ori2new)={len(qid_ori2new)} != {data_config['num_q']}"
    cid_ori2new = keyid2idx["concepts"]
    assert len(cid_ori2new) == data_config[
        "num_c"], f"len(cid_ori2new)={len(cid_ori2new)} != {data_config['num_c']}"

    return qid_ori2new, cid_ori2new

def get_auc_acc_datas_from_predictiong_logs(dir_saved_model: str):
    # 读取 dir_saved_model 目录下的每个子目录（子目录名称字符串中包含"_qid_"子串）
    all_datas = defaultdict(list)

    # 先筛选指定条件的子目录
    def is_dir_include_files(dir):
        fnames = ['config.json', 'predicting', 'log']
        str_test = '&'.join(os.listdir(dir))
        for f in fnames:
            if f not in str_test:
                return False

        return True

    sub_dirs = []
    for sub_dir in os.listdir(dir_saved_model):
        dir0 = os.path.join(dir_saved_model, sub_dir)
        if os.path.isdir(dir0) and is_dir_include_files(dir0):
            sub_dirs.append(dir0)

    num_process = 0
    for sub_dir in tqdm(sub_dirs, desc="Processing"):
        with open(os.path.join(sub_dir, "config.json"), "r") as f:
            config = json.load(f)
        knames_needed1 = ['model_name', 'dataset_name', 'fold']
        args = {}
        for _, d0 in config.items():
            for k, v in d0.items():
                if k in knames_needed1:
                    args[k] = v

        all_preds_datas = {}
        # 读取子目录sub_dir下的predicting.log文件
        file_name = [fname for fname in os.listdir(sub_dir) if
                     fname.endswith(".log") and fname.startswith("predicting")]
        log_file_path = os.path.join(sub_dir, "predicting.log")
        if len(file_name) > 0:
            # file_name 列表可能有多个 predicting_YYMMDD-hhmmss.log 文件
            file_name = sorted(file_name)[-1]
            log_file_path = os.path.join(sub_dir, file_name)
            num_process += 1
            with open(log_file_path, "r") as f:
                lines = f.readlines()
                if lines:
                    # 在倒数前5行中找以"{'testauc'"开头的行
                    for line in lines[-5:]:
                        if line.startswith("{'testauc'"):
                            all_preds_datas = eval(line.strip())

        # knames_needed2 = ['testauc', 'testacc', 'window_testauc', 'window_testacc', 
        #                  'oriaucconcepts', 'oriauclate_mean', 'oriauclate_vote', 
        #                  'oriauclate_all', 'oriaccconcepts',  'oriacclate_vote',
        #                    'oriacclate_all', 'windowaucconcepts', 'windowauclate_mean', 
        #                    'windowauclate_vote', 'windowauclate_all', 'windowaccconcepts', 
        #                    'windowacclate_mean', 'windowacclate_vote', 'windowacclate_all']
        knames_needed2 = ['testauc', 'testacc',
                          'window_testauc', 'window_testacc',
                          'oriauclate_mean', 'windowauclate_mean',
                          'oriacclate_mean', 'windowacclate_mean']
        one_datas = {k: v for k, v in args.items()}
        for kname in knames_needed2:
            if kname in all_preds_datas:
                one_datas[kname] = all_preds_datas[kname]
            else:  # 有些 模型或者数据集 是没有 late_mean 相关的数据
                one_datas[kname] = None

        all_datas[args['dataset_name']].append(one_datas)

    #  all_datas  中每个数据集的结果保存到 csv 文件中
    for dataset_name, datas in all_datas.items():
        df = pd.DataFrame(datas)
        # 数据行 排序
        df.sort_values(by=['model_name', 'fold'], inplace=True)
        # 保存到 dir_saved_model目录下的predicting_datas.csv文件中
        file_saved = os.path.join(dir_saved_model, f"{dataset_name}_predicting_datas.csv")
        df.to_csv(file_saved, index=False)

        df_group_by_model = df.groupby(['model_name'])
        datas_mean = []
        for model_name, df_model in df_group_by_model:
            mname = model_name[0]
            df_model.drop(columns=['model_name', 'dataset_name', 'fold'], inplace=True)
            mean_data = df_model.mean().to_dict()
            mean_data['model_name'] = mname
            mean_data['dataset_name'] = dataset_name

            datas_mean.append(mean_data)

        file_saved = file_saved.replace(".csv", "_mean.csv")
        df_mean = pd.DataFrame(datas_mean)
        df_mean.to_csv(file_saved, index=False)

    return all_datas


if __name__ == "__main__":
    # get_predicting_datas_from_log("/home3/junkang/JJK/text2multimodal_for_kt/scripts_training2testing/saved_model_DBE_KT22")
    pass
    import argparse

    dir_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_saved_model", type=str,
                        default="")
    parser.add_argument("--dataset_name", type=str, default="DBE_KT22")
    parser.add_argument("--data_config", type=str,
                        default=os.path.join(dir_project, "my_configs", "data_config.json"))
    args = parser.parse_args()
    if args.dir_saved_model != "":
        get_auc_acc_datas_from_predictiong_logs(args.dir_saved_model)
