import sqlite3
from tqdm import tqdm
import pickle

DATA_BASE = '/Users/zhengquanwei/ChatBotDB.db'


# 处理多线程
conn = sqlite3.connect(DATA_BASE, timeout=10, check_same_thread=False)

cur = conn.cursor()


def insert_chat_record(question, answer):
    sql_text = '''INSERT INTO chats (QUESTION, ANSWER) VALUES('%s', '%s');''' % (question, answer)
    cur.execute(sql_text)
    conn.commit()


def insert_chat_series(chat_series):
    """
        一次性插入多条数据
    """
    cur.executemany('INSERT INTO chats (QUESTION, ANSWER) VALUES (?,?) ', chat_series)
    conn.commit()


def copy_into_db(train_data_path):
    """
        将训练数据批量录入到数据库中
    """
    with open(train_data_path, 'r',encoding='utf-8') as f:
        data = f.read()
    # 需要区分linux和windows环境下的换行符
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    # 遍历问答对
    series_to_insert = []
    for index, dialogue in enumerate(tqdm(train_data)):
        # 限制条数
        if index > 10:
            insert_chat_series(series_to_insert)
            # 清空队列
            series_to_insert.clear()
            return
        # 200条插入一次
        if index % 4 == 0:
            insert_chat_series(series_to_insert)
            # 清空队列
            series_to_insert.clear()
        if "\r\n" in data:
            utterances = dialogue.split("\r\n")
        else:
            utterances = dialogue.split("\n")
        question = utterances[0].strip()
        try:
            answer = utterances[1].strip()
        except Exception as e:
            print(e)
        # 插入一条数据到队列中
        series_to_insert.append((question, answer))

    # 如果循环结束队列中还有数据剩余，则直接进行插入
    if len(series_to_insert) > 0:
        insert_chat_series(series_to_insert)
        series_to_insert.clear()


def query_by_question(question):
    """
        通过问题查询答案
    """
    sql_text = '''SELECT * FROM chats WHERE QUESTION LIKE '%s' ''' % question
    cur.execute(sql_text)
    # tuple 类型 (id, question, answer)
    res = cur.fetchone()
    if res is not None:
        # print('找到问题答案：%s' % res[2])
        return res[2]
    # print('数据库未找到问题的答案')
    return None


if __name__ == '__main__':
   q = '胃炎能买医疗险?'
   a = '胃炎能买保险与否，其实医疗险和重疾险的核保结论有一些不一样（1）医疗险：通常考虑责任除外承保（2）重疾险：现症——延期/加费治疗后已治愈或症状已消除（HP阴性）——可标准体承保。一些互联网重疾险假设胃溃疡已经治愈6个月以上，并且幽门螺旋杆菌也被治愈了，就可以直接投保重疾险，标准体承保的产品非常多。除了胃炎，还有胃息肉，就是胃内表面长出来的一个“肉疙瘩”，大多是在胃镜检查中发现，医疗险的核保会比重疾险严格很多，2年以上被治愈的胃部息肉才有可能考虑标准体承保，否则责任除外；但是很多重疾险，只要是胃部息肉为良性，正常投保的也不少。'
   # insert_chat_record(q, a)

   # query_by_question(q)

   q2 = '儿童买什么保险?'
   # query_by_question(q2)

   # copy_into_db('./insuranceQa/qa_6000-7000.txt')


   chat_series = [
    (
      '你好',
      '你好，我是练习两周半的 AI 聊天机器人！请问有什么可以帮你'
    ),
    ('hello',
     '你好，我是练习两周半的 AI 聊天机器人！请问有什么可以帮你'
    ),
    (
      'hi',
      '你好，我是练习两周半的 AI 聊天机器人！请问有什么可以帮你'
    ),
       (
           '你是',
           '你好，我是练习两周半的 AI 聊天机器人！请问有什么可以帮你'
       )
   ]

   insert_chat_series(chat_series)