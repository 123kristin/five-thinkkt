import os
import pandas as pd
from .utils import sta_infos, write_txt

def read_data_from_csv(dir_path, writef):
    transaction_df = pd.read_csv(os.path.join(dir_path, "Transaction.csv"))
    cols_needed = ['start_time', 'end_time', 'answer_state', 'student_id', 'question_id']
    transaction_df = transaction_df[cols_needed]

    transaction_df['start_time'] = pd.to_datetime(transaction_df['start_time'], errors='coerce',
                                                  utc=True).dt.tz_convert(None)
    transaction_df['end_time'] = pd.to_datetime(transaction_df['end_time'], errors='coerce', utc=True).dt.tz_convert(
        None)

    transaction_df['time_taken'] = (transaction_df['end_time'] - transaction_df['start_time'])
    transaction_df = transaction_df[['student_id', 'question_id', 'answer_state', 'start_time', 'time_taken']]
    # 将 start_time	time_taken 分别转换为时间戳、秒数
    transaction_df['timestamp'] = transaction_df['start_time'].astype('int64') // 10 ** 9
    transaction_df['time_taken'] = transaction_df['time_taken'].dt.total_seconds().astype('int64')
    df = transaction_df.drop(columns=['start_time'])  # 删除 start_time

    # 加入 concepts_id 列
    question_kc_df = pd.read_csv(os.path.join(dir_path, "Question_KC_Relationships.csv"))
    list_concepts = []
    for i in range(df.shape[0]):
        question_id = df.loc[i, "question_id"]
        concepts = question_kc_df[question_kc_df["question_id"] == question_id]["knowledgecomponent_id"].tolist()
        concepts_str = '_'.join(map(str, concepts))
        list_concepts.append(concepts_str)
    df["concepts_id"] = list_concepts

    KEYS = ["student_id", "question_id"]

    stares = []

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["student_id", "question_id", "answer_state", "timestamp"])

    df['answer_state'] = df['answer_state'].astype('Int8')  # 是为了转为0或1的数字（原来是false/true）

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    data = []
    ui_df = df.groupby('student_id', sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        curdf = curdf.sort_values(by=["timestamp", "index"])

        # problem -> concept
        concepts = curdf["concepts_id"].astype(str)
        responses = curdf["answer_state"].astype(str)
        timestamps = curdf["timestamp"].astype(str)
        questions = curdf["question_id"].astype(str)
        usetimes = curdf["time_taken"].astype(str)
        uids = [str(uid), str(len(responses))]
        data.append([uids, questions, concepts, responses, timestamps, usetimes])

    write_txt(writef, data)