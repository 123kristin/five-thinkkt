import os

import argparse
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import process_raw_data

dir_current = os.path.dirname(os.path.abspath(__file__))
dir_project = os.path.dirname(dir_current)
dir_project = os.path.dirname(dir_project)

# 字典 dname2paths包含不同数据集的路径，其中key是数据集的名称，value是对应数据集的路径
dname2paths = {
    "nips_task34": "data/Eedi/data/train_data/train_task_3_4.csv",
    "DBE_KT22": "data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv",
    "MOOCRadar": "data/MOOCRadar/student-problem-coarse.json"
    
}

dname2paths = {
    k : os.path.join(dir_project, v) for k, v in dname2paths.items()  
}

# 配置文件 data_config.json 的路径
configf = "my_configs/data_config.json"
configf = os.path.join(dir_project, configf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="nips_task34")
    parser.add_argument("-f","--file_path", type=str, default="")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)

    # process raw data
    if args.dataset_name=="peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}")
    # split
    os.system("rm " + dname + "/*.pkl")

    #for concept level model
    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)

    #for question level model
    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)

