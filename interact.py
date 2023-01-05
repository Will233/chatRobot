
import faulthandler
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import database.db as db
# websocket server
from flask import Flask, render_template, request, url_for
from flask_socketio import SocketIO, emit
import asyncio
import websockets
faulthandler.enable()

PAD = '[PAD]'
pad_id = 0

region_words = ['保险', '人寿保险', '平安', '医疗保险', '汽车保险', '医疗险', '家财险', '意外险', '保险公司', '重疾险', '保险费用',
                '人寿', '寿险', '保险费', '险', '人身保险', '理赔', '平安保险', '保费', '车险', '保险金', '被保险人', '投保人',
                '保险费率', '财产保险', '保单', '续保', '投保', '核保', '医保',
                # 特定保险begin
                '综合意外', '鹏城保', '百万家财', '水滴保', '福禄鑫尊', '国寿瑞鑫', '鑫裕金', '鑫尊宝', '国寿福', '同佑e生', '金佑人生',
                '金福合家欢', '重庆渝惠保', '国寿康宁', '泰康贴心保', '新冠隔离津贴', '火车隔离津贴', '外卖准时宝', '泰康', '悟空保', '南充充惠保',
                '春城惠民保', '太平福禄御禧', '太平洋金佑', '国华金如意', '医疗保障', '医疗保健',
                # 特定保险end
                # 特定术语begin
                '告知义务', '说明义务', '现金价值', '犹豫期', '宽限期', '条款',
                # 特定术语end
                '重大疾病',
                # '年金',
                # '退休计划',
                # '养老金',
                '免赔', '保障期', '退保', '保额', '受益人', '可以保', '能保么',
                ]


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=4, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False, help='对话模型路径')
    parser.add_argument('--config_path', default='model/epoch40', type=str, required=False, help='对话模型配置路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.5, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=255, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=1, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # start_time = time.time()
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    # end_time = time.time()
    # print('用时：%d s' % (end_time - start_time))
    return logits


#
def main():
    args = set_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    config = GPT2Config.from_json_file(args.config_path)
    print(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)
    logger.info(f'使用模型 {args.model_path} 对应配置 {args.config_path}')
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            user_input = input("user:")
            # text = "你好"
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(user_input))
            response = []  # 根据context，生成的response
            # 查询数据库
            # answer = db.query_by_question(user_input)
            # if answer is not None:
            #     print("chatbot: %s" % answer)
            #     response.append(answer)
            #     continue
            text_ids = tokenizer.encode(user_input, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)

            # print("chatbot:", end='')
            response_text = ''
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
                text = tokenizer.convert_ids_to_tokens(response)
                # response_text = response_text.join(text)
                # print(text[-1], end='')
            history.append(response)
            # 加入到数据缓存中
            # print(user_input)
            # print(response)
            # db.insert_chat_record(user_input, response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot:" + "".join(text))
            # 保存历史答案
            # db.insert_chat_record(user_input, "".join(text))
            # print(end='\n')

            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break

# websocket + model
def create_app(args):
    # app server
    app = Flask(__name__)
    # app.config['SECRET_KEY'] = 'secret!'
    socketio = SocketIO(app, cors_allowed_origins='*')
    # 模型配置
    logger = create_logger(args)
    history = []
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    config = GPT2Config.from_json_file(args.config_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)
    logger.info(f'使用模型 {args.model_path} 对应配置 {args.config_path}')
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))

    # 路由配置
    @app.route("/")
    def index():
        return render_template("index.html")

    @socketio.on('ask', namespace='/test')
    def bot_answer(message):
        # text = "你好"
        print(message)
        user_input = message
        response = []  # 根据context，生成的response
        # 1、查询数据库
        answer = db.query_by_question(user_input)
        if answer is not None:
            print("chatbot: %s" % answer)
            # response.append(answer)
            # continue
            emit('answer-finish', answer)
            return
        else:
            # 1、判断领域
            in_region = False
            for region_word in region_words:
                if not user_input.__contains__(region_word):
                    pass
                else:
                    in_region = True
                    break
            if not in_region:
                response_text = "这方面的问题，我不太明白，要不您换个问法再试试，或许我就能明白啦！"
                emit('answer-finish', response_text)
                return
            # 2、走模型
            text_ids = tokenizer.encode(user_input, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)

            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    # 结束回答
                    emit('answer-finish', '[END]')
                    break
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                current_token = next_token.item()
                response.append(current_token)
                current_text = tokenizer.convert_ids_to_tokens([current_token])
                # 回答
                emit('answer-ing', current_text)
            # 超过最大值
            history.append(response)
            emit('answer-finish', '[END]')
    @socketio.on('my event', namespace='/test')
    def test_message(message):
        emit('my response', {'data': message['data']})

    @socketio.on('my broadcast event', namespace='/test')
    def test_message(message):
        emit('my response', {'data': message['data']}, broadcast=True)

    @socketio.on('connect', namespace='/test')
    def test_connect():
        emit('my response', {'data': 'Connected'})

    @socketio.on('disconnect', namespace='/test')
    def test_disconnect():
        print('Client disconnected')

    socketio.run(app, '0.0.0.0', '8080')


class ModelArgs:
    device = '0'
    topk = 4
    topp = 0
    log_path = 'data/interact.log'
    vocab_path = 'vocab/vocab.txt'
    model_path = './model/quality-epoch-64/pytorch_model_quality_189.bin'
    config_path = './model/quality-epoch-64/config.json'
    save_samples_path = 'sample/'
    repetition_penalty = 1.5
    max_len = 255
    max_history_len = 1
    temperature = 1
    def __init__(self, model_path, max_len, repetition_penalty):
        self.max_len = max_len
        self.model_path = model_path
        self.repetition_penalty = repetition_penalty


if __name__ == '__main__':
    # main()
    # 运行服务
    args = ModelArgs('./model/quality-epoch-64/pytorch_model_quality_189.bin', 255, 1.5)
    create_app(args)
