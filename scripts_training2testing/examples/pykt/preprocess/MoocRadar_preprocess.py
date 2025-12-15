import os
import pandas as pd
import json
from tqdm import tqdm

try:
    from .utils import sta_infos, write_txt
except:
    from utils import sta_infos, write_txt

from datetime import datetime


def read_data_from_json(json_path, writef):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas_list = json.load(f)
    
    dir_path = os.path.dirname(json_path)
    
    dict_problem_id2concepts = {}
    with open(os.path.join(dir_path, "problem.json"), 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            dict_problem_id2concepts[problem['problem_id']] = "_".join(problem['concepts'])
    
    ## datas_list 的每个元素是一个只有一个键值对的字典{ seq:stu_logs_list }, 也即学生答题序列数据
    ## stu_logs_list 的每个元素是一个字典，键：['log_id', 'problem_id', 'user_id', 'is_correct', 'attempts', 'score', 'submit_time', 'skill_id', 'course_id']

    KEYS = ["user_id", "problem_id"]

    data = []

    for ui in tqdm(datas_list):
        stu_logs_list = ui['seq']
        uid = stu_logs_list[0]['user_id']
        num_logs = len(stu_logs_list)

        concepts = []
        responses = []
        timestamps = []
        questions = []

        for log in stu_logs_list:
            questions.append(str(log['problem_id']))
            concepts.append(dict_problem_id2concepts[log['problem_id']])
            responses.append(str(log['is_correct']))
            #将 log['submit_time'] 转为 时间戳格式（原格式：xxxx-xx-xx xx:xx:xx）
            timestamp = datetime.strptime(log['submit_time'], "%Y-%m-%d %H:%M:%S").timestamp()
            timestamps.append(str(int(timestamp)))

        usetimes = ["NA"]
        uids = [str(uid), str(num_logs)]
        data.append([uids, questions, concepts, responses, timestamps, usetimes])

    write_txt(writef, data)


if __name__ == '__main__':
    read_data_from_json("/home3/junkang/JJK/sparseMoE-MKT/datas_files/MOOCRadar/student-problem-coarse.json", "/home3/junkang/JJK/sparseMoE-MKT/datas_files/MOOCRadar/student-problem-coarse.txt")