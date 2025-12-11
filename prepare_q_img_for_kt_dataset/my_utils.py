import os
import json

def get_qid_ori2new(data_config: dict):
    dpath = data_config["dpath"]
    file_keyid2idx = os.path.join(dpath, "keyid2idx.json")
    if not os.path.exists(file_keyid2idx):  # 有些数据集不用重映射，比如XES3G5M
        return None

    with open(file_keyid2idx, "r") as f:
        keyid2idx = json.load(f)
    qid_ori2new = keyid2idx["questions"]
    assert len(qid_ori2new) == data_config[
        "num_q"], f"len(qid_ori2new)={len(qid_ori2new)} != {data_config['num_q']}"

    return qid_ori2new