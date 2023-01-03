import insuranceqa_data as insuranceqa
import pandas as pd
# 使用时需要安装 `insuranceqa_data`
# 如果下载不下来，可以手动下载数据集，存储到本地， 不含corpus目录解压到`insuranceqa_data`包的文件夹下
# 链接: https://pan.baidu.com/s/1GTML6qJuqpK_gT4-ShQPKw 提取码: q3u3

# 给insuranceqa_data/__init__.py 补充若干方法
# def load_train():
#     TRAIN_DATA = os.path.join(curdir, 'pool', 'train.json.gz')
#     return load(TRAIN_DATA)
#
# def load_test():
#     TEST_DATA = os.path.join(curdir, 'pool', 'test.json.gz')
#     return load(TEST_DATA)
#
# def load_valid():
#     TEST_DATA = os.path.join(curdir, 'pool', 'valid.json.gz')
#     return load(TEST_DATA)

test_data_path = './data/corpus_test_data.txt'
valid_data_path = './data/corpus_valid_data.txt'
train_data_path = './data/corpus_train_data.txt'

test_data_en_path = './data/corpus_test_data_zh.txt'
valid_data_en_path = './data/corpus_valid_data_zh.txt'
train_data_en_path = './data/corpus_train_data_zh.txt'
# vocab_data = insuranceqa.load_pairs_vocab()
vocab_data = {}
def get_real_words(id_array):

    sentence = ''
    for word_id in id_array:
        sentence += vocab_data['id2word'][str(word_id)]
    return sentence


def extract_data(save_path, data_type):
    if data_type == 'train':
        datas = insuranceqa.load_train()
    elif data_type == 'test':
        datas = insuranceqa.load_test()
    else:
        datas = insuranceqa.load_valid()
    answers = insuranceqa.load_answers()
    with open(save_path, 'a', encoding='utf-8') as f:
        for qid in datas:
            question = datas[qid]['zh']
            right_answer_id = datas[qid]['answers'][0]
            answer = answers[right_answer_id]['zh']
            f.write(question + '\n')
            f.write(answer[1:] + '\n')
            f.write('\n')
            # print(question + '\n')
            # print(answer[1:] + '\n')


def extract_baoxian_filter():
    df = pd.read_csv('baoxianzhidao_filter.csv')
    df2 = df[df['is_best']==1]
    print(df2.size)
    with open('baoxianzhidao_with_right.txt', 'a', encoding='utf-8') as f:
        for i in range(0, len(df)):
            row = df.iloc[i]
            if row['is_best'] == 1:
                f.write(row['title'] + '\n')
                f.write(row['reply'] + '\n')
                f.write('\n')
            else:
                print('不是最佳答案')


if __name__ == '__main__':
    extract_baoxian_filter()
    # extract_data(train_data_en_path, 'train')
    # extract_data(test_data_en_path, 'test')
    # extract_data(valid_data_en_path, 'valid')
    # train_data = insuranceqa.load_pairs_train()
    # test_data = insuranceqa.load_pairs_test()
    # valid_data = insuranceqa.load_pairs_valid()
    # vocab_data = insuranceqa.load_pairs_vocab()
    # test data
    # with open(test_data_path, 'a', encoding='utf-8') as f:
    #     for x in test_data:
    #         if x['label'][0] == 1:
    #             question = get_real_words(x['question'])
    #             utterance = get_real_words(x['utterance'])
    #             print(x)
                # f.write(question + '\n')
                # f.write(utterance + '\n')
                # f.write('\n')

    # train data
    # with open(train_data_path, 'a', encoding='utf-8') as f:
    #     for x in train_data:
    #         if x['label'][0] == 1:
    #             question = get_real_words(x['question'])
    #             utterance = get_real_words(x['utterance'])
    #             f.write(question + '\n')
    #             f.write(utterance + '\n')
    #             f.write('\n')
    # # valid data
    # with open(valid_data_path, 'a', encoding='utf-8') as f:
    #     for x in valid_data:
    #         if x['label'][0] == 1:
    #             question = get_real_words(x['question'])
    #             utterance = get_real_words(x['utterance'])
    #
    #             f.write(question + '\n')
    #             f.write(utterance + '\n')
    #             f.write('\n')
