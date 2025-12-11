try:
    from q_imgs_obtain.q_materials_to_html import text2html
    from q_imgs_obtain.html2img import html_to_image_imgkit
    from my_utils import get_qid_ori2new
except:
    from .q_imgs_obtain.q_materials_to_html import text2html
    from .q_imgs_obtain.html2img import html_to_image_imgkit
    from .my_utils import get_qid_ori2new

import os
import json
import pandas as pd
from tqdm import tqdm


def prepare_q_imgs(dataset_name, datas_dir, file_configf, repreprocess=False):
    """
        获取 数据集中的需要的问题元数据, 包括id及其对应的内容(图文：图片路径列表、文本)
        :param datas_dir: 数据集根目录
        :param dataset_name: 数据集名称
        :param file_configf: 配置文件路径
        :return:

    """
    with open(file_configf, "r") as f:
        dict_data_config = json.load(f)
        data_config = dict_data_config[dataset_name]

    dict_qid2qcontent = {}

    if "XES3G5M" in dataset_name:
        datas_dir = os.path.join(datas_dir, "metadata")
        with open(os.path.join(datas_dir, "questions.json")) as f:
            dict_ques = json.load(f)
        with open(os.path.join(datas_dir, "kc_routes_map.json")) as f:
            map_cid2content = json.load(f)

        num_q = 0
        for qid, que in tqdm(dict_ques.items(), desc="Preparing Que contents..."):
            q_text = que["content"]
            imgname2path = {}
            for img_id in range(10):
                img_key = f"question_{qid}-image_{img_id}"
                img_path = os.path.join(datas_dir, "images", f"{img_key}.png")
                if os.path.exists(img_path):
                    q_text = q_text.replace(img_key, f"\n<{img_key}>")
                    imgname2path[img_key] = img_path
                else:
                    break

            if que['type'] != "填空":
                text_options = ""
                for k, v in que['options'].items():
                    text_options += f"\n{k}. {v}"
                q_text += text_options

            dict_qid2qcontent[qid] = (q_text, imgname2path)
            num_q += 1
        print(f"num_q={num_q}")

    elif "DBE_KT22" in dataset_name:
        qid_ori2new = get_qid_ori2new(data_config)

        question_df = pd.read_csv(os.path.join(datas_dir, "Questions.csv"))[['id', 'question_text']]
        que_choices_df = pd.read_csv(os.path.join(datas_dir, "Question_Choices.csv"))[["choice_text", "question_id"]]
        df_kcs = pd.read_csv(os.path.join(datas_dir, "KCs.csv"))  # ["id","name","description"]

        # 将所有的 id 转为 str 类型
        question_df['id'] = question_df['id'].astype(str)
        que_choices_df['question_id'] = que_choices_df['question_id'].astype(str)
        df_kcs['id'] = df_kcs['id'].astype(str)

        num_q = 0
        for _, q_row in tqdm(question_df.iterrows(), desc="Preparing Ques content..."):
            qid = q_row['id']
            if qid not in qid_ori2new:
                continue

            q_text = q_row['question_text']

            choices_list = que_choices_df[que_choices_df['question_id'] == qid]['choice_text'].tolist()
            if len(choices_list) > 0:
                option_id = ord('A')  # 从A开始
                text_choices = ''
                for i in range(len(choices_list)):
                    text_choices += f"\n{chr(option_id + i)}. {choices_list[i]}"
                q_text += text_choices

            dict_qid2qcontent[qid_ori2new[qid]] = (q_text, dict())
            num_q += 1
        print(f"num_q={num_q}")
    
    elif "NIPS_task34" in dataset_name:
        qid_ori2new = get_qid_ori2new(data_config)
        # 注意 这个数据集的题目全为图片表示，一张图片即为一个问题
        num_q = 0
        q_imgs_dir = os.path.join(datas_dir, "images")
        # 遍历所有图片
        for qimg in tqdm(os.listdir(q_imgs_dir), desc="Preparing Ques content..."):
            qid = qimg.split(".")[0]
            qpath = os.path.join(q_imgs_dir, qimg)
            qcontent = qpath
            dict_qid2qcontent[qid_ori2new[qid]] = qcontent
            num_q += 1
        
        print(f"num_q={num_q}")

    elif "MOOCRadar" in dataset_name:
        pass
    else:
        raise ValueError("Invalid dataset name.")

    # return {int(qid):qcontent for qid, qcontent in dict_qid2qcontent.items()}

    dict_qid2qimg_path = {}
    dir_saved = os.path.join(datas_dir, "q_imgs")
    os.makedirs(dir_saved, exist_ok=True)

    if "NIPS_task34" in dataset_name:
        dict_qid2qimg_path = {int(qid): qcontent for qid, qcontent in dict_qid2qcontent.items()}
    else:
        for qid, (q_text, imgname2path) in tqdm(dict_qid2qcontent.items(), desc="Preparing Ques imgs..."):
            q_html = text2html(q_text, imgname2path, None)
            img_path = os.path.join(dir_saved, f"{qid}.jpg")
            if repreprocess or not os.path.exists(img_path):  # 若需要重新处理或本来就没处理过
                html_to_image_imgkit(q_html, img_path)

            dict_qid2qimg_path[int(qid)] = img_path
    
    return dict_qid2qimg_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="DBE_KT22")
    parser.add_argument("--datas_dir", type=str, default="")
    parser.add_argument("--file_configf", type=str, default="")
    parser.add_argument("--repreprocess", type=int, default=0)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    datas_dir = args.datas_dir
    file_configf = args.file_configf
    repreprocess = args.repreprocess

    from config_base import map_dataset_name2datas_dir, file_configf

    if datas_dir == "":
        datas_dir = map_dataset_name2datas_dir[dataset_name]

    if file_configf == "":
        file_configf = file_configf

    dict_qid2qimg_path = prepare_q_imgs(dataset_name, datas_dir, file_configf, repreprocess)