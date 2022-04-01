import copy
import gc
import logging
import nturl2path
import os

# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
from collections import Counter
import time
from tracemalloc import start
import unicodedata
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertModel, BertTokenizerFast, \
    get_polynomial_decay_schedule_with_warmup, AutoConfig
from torch import cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # 0,1,2,3 for four gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
start_time = time.localtime(time.time())
start_time_str = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'
# VERSION FOR SAVING MODEL WEIGHTS
VER = 1

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = None
LOAD_DATA_FROM = None
# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = './prev_trained_model/uer-large'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'uer_large'
log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                               datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger()
if LOAD_MODEL_FROM is None:
    OUTPUT_EVAL_FILE = os.path.join('./outputs/logs', 'eval_results_' + start_time_str + '.txt')
    logfile = './outputs/logs/' + MODEL_NAME + '_log_' + start_time_str + '.log'

    logging.basicConfig(level=logging.DEBUG)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.info(MODEL_NAME)

torch.set_printoptions(threshold=np.inf)
#################################################################
# config
config = {'model_name': MODEL_NAME,
          'max_length': 128,
          'train_batch_size': 32,
          'valid_batch_size': 64,
          'epochs': 25,
          'learning_rate_for_bert': 5e-5,
          'learning_rate_for_others': 2e-3,
          'weight_decay': 1e-6,
          'max_grad_norm': 10,
          'device': 'cuda:0' if cuda.is_available() else 'cpu'}
logger.info(config)
# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True

###############################################################
# 转化为ner标签
# LOAD_TOKENS_FROM 为存储ner标签csv的路径，处理一次，以后就不用处理了
# Ner标签采用BIO标记法
# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['1', '2', '3', '4', '5', '6',
                 '7', '8', '9', '10', '11', '12', '13',
                 '14', '15', '16', '17', '18', '19',
                 '20', '21', '22', '23', '24', '25', '26',
                 '28', '29', '30', '31', '32', '33',
                 '34', '35', '36', '37', '38', '39', '40',
                 '41', '42', '43', '44', '46', '47',
                 '48', '49', '50', '51', '52', '53', '54']
# 存储标签与index之间的映射
labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}


################################################################
# 读取训练集标签
def get_entity(seq):
    """
        seq: [seq_length]
        return:[[entity, start, end], ...]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


'''
def read_text(input_file):
    datalist = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        labels = []

        for line in lines:
            if line == '\n':
                # 转化为一句
                text = ''.join(text)
                entity_labels = get_entity(labels)

                if text == '':
                    continue

                datalist.append({
                    'words': text,
                    'labels': entity_labels
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                labels.append(label)
    return datalist
'''


def read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": ''.join(words), "labels": get_entity(labels)})

                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                if splits[0] == '':
                    words.append(' ')
                else:
                    words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": ''.join(words), "labels": get_entity(labels)})
    return lines


def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


def recover_bert_token(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def get_token_mapping(text, tokens, is_mapping_index=True):
    """给出原始的text和tokenize后的tokens的映射关系"""
    raw_text = copy.deepcopy(text)
    text = text.lower()
    additional_special_tokens = []
    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        ch = unicodedata.normalize('NFD', ch)
        ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        token = token.lower()
        if token == '[unk]' or token in additional_special_tokens:
            if is_mapping_index:
                token_mapping.append(char_mapping[offset:offset + 1])
            else:
                token_mapping.append(raw_text[offset:offset + 1])
            offset = offset + 1
        elif _is_special(token):
            # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            token_mapping.append([])
        else:
            token = recover_bert_token(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            if is_mapping_index:
                token_mapping.append(char_mapping[start:end])
            else:
                token_mapping.append(raw_text[start:end])
            offset = end

    return token_mapping


def get_ent2token_spans(tokenizer, text, entity_list):
    """实体列表转为token_spans
    Args:
        text (str): 原始文本
        entity_list (list): [(ent_type, start, end),(ent_type, start, end)...]
    """
    ent2token_spans = []
    n = 0
    total = 0
    inputs = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    token2char_span_mapping = inputs["offset_mapping"]
    text2tokens = tokenizer.tokenize(text, add_special_tokens=True)

    token_mapping = get_token_mapping(text, text2tokens)
    start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}
    for ent_span in entity_list:
        total += 1

        ent = text[ent_span[1]:ent_span[2] + 1]
        start_index, end_index = -1, -1
        if ent_span[1] in start_mapping and ent_span[2] in end_mapping:
            start_index = start_mapping[ent_span[1]]
            end_index = end_mapping[ent_span[2]]
            if start_index > end_index or ent_span == '':
                n += 1
                continue
            ent2token_spans.append([ent_span[0], start_index, end_index])

            '''
        ent = text[ent_span[1]:ent_span[2] + 1]
        ent2token = tokenizer.tokenize(ent, add_special_tokens=False)
        # 寻找ent的token_span
        token_start_indexs = []
        token_end_indexs = []
        for i, v in enumerate(text2tokens):
            v1 = v.replace('#', '')
            e1 = ent2token[0].replace('#', '')
            if (len(v1) <= len(e1) and v1 == e1[:len(v1)]):# or (len(v1) > len(e1) and v1[:len(e1)] == e1):
                token_start_indexs.append(i)
        for i, v in enumerate(text2tokens):
            v1 = v.replace('#', '')
            e1 = ent2token[-1].replace('#', '')
            if (len(v1) <= len(e1) and v1 == e1[-len(v1):]):# or (len(v1) > len(e1) and v1[-len(e1):] == e1):
                token_end_indexs.append(i)
        # token_start_indexs = [i for i, v in enumerate(text2tokens) if v.replace('#', '') == ent2token[0].replace('#', '')]
        # token_end_indexs = [i for i, v in enumerate(text2tokens) if v.replace('#', '') == ent2token[-1].replace('#', '')]
        # print(token_start_indexs, token_end_indexs)

        token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[1], token_start_indexs))
        token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[2],
                                      token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间
        # print(token_start_index, token_end_index)
        if len(token_start_index) == 0 or len(token_end_index) == 0:

            print(token_start_indexs, token_end_indexs)
            print(token_start_index, token_end_index)
            print(ent2token, ent_span)
            print(text2tokens)
            print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
            '''
        else:
            n += 1
            continue
        # token_span = (ent_span[0], token_start_index[0], token_end_index[0])
        # ent2token_spans.append(token_span)

    return ent2token_spans, n, total


def preprocessor(df, tokenizer):
    # GET TEXT AND WORD LABELS
    total, drop = 0, 0
    for index in tqdm(range(len(df))):
        data = df[index]
        text = data['words']
        word_labels = data['labels']
        ent2token_spans, n1, n2 = get_ent2token_spans(
            tokenizer, text, word_labels
        )
        drop += n1
        total += n2
        # TOKENIZE TEXT：生成input_ids, input_mask
        input_ids = tokenizer(text)["input_ids"]
        input_mask = [1] * len(input_ids)
        # CONVERT TO TORCH TENSORS
        df[index] = {'text': text,
                     'input_ids': input_ids,
                     'attention_mask': input_mask,
                     'labels': ent2token_spans}
    print(total, drop)
    return df


tokenizer = BertTokenizerFast.from_pretrained(DOWNLOADED_MODEL_PATH, do_lower_case=True)

#########################################################################
# 定义dataset

LABEL_ALL_SUBTOKENS = True


class dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        # print(self.data[index])
        return self.data[index]

    def __len__(self):
        return self.len


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        maxlen = 160
        maxlen = max([len(idx) for idx in input_ids])
        labels = torch.zeros((len(batch), len(output_labels), maxlen, maxlen))
        for sample in range(len(batch)):
            for ent, start, end in batch[sample]['labels']:
                labels[sample, labels_to_ids[ent], start, end] = 1
        # 补全
        input_ids = [s + [self.tokenizer.pad_token_id] * (maxlen - len(s)) for s in input_ids]
        attention_mask = [s + [0] * (maxlen - len(s)) for s in attention_mask]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_id = torch.zeros((len(batch), maxlen))
        return input_ids, attention_mask, labels, segment_id


#################################################################

train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 6,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 6,
               'pin_memory': True
               }
collate = Collate(tokenizer)
# 读取训练文件
if LOAD_MODEL_FROM is None:
    print('preprocess train')
    train_text = read_text('./datasets/JDNER/train_all.txt')
    train_df = preprocessor(train_text, tokenizer)
    train_set = dataset(train_df)
    # torch.save(train_df, '/home/zbg/Reinforcement/强化学习算法2021/JDNER/datasets/JDNER/train')
    print("TRAIN Dataset: {}".format(len(train_df)))

    # 读取开发集文本
    print('preprocess dev')
    dev_text = read_text('./datasets/JDNER/dev.txt')
    dev_df = preprocessor(dev_text, tokenizer)
    dev_set = dataset(dev_df)
    # torch.save(dev_df, '/home/zbg/Reinforcement/强化学习算法2021/JDNER/datasets/JDNER/dev')
    print("DEV Dataset: {}".format(len(dev_df)))

    train_dataloader = DataLoader(train_set, **train_params, collate_fn=collate)
    dev_dataloader = DataLoader(dev_set, **test_params, collate_fn=collate)


#########################################################################
# 模型
def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class GlobalPointer(nn.Module):
    def __init__(self, model_path, ent_type_size, inner_dim, RoPE=True, hidden_size=1024):
        super().__init__()
        # self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        )

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, output_dim, merge_mode, inputs):
        input_shape = inputs.shape
        _, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, output_dim))

        if merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif merge_mode == 'zero':
            return embeddings.to(inputs.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-1]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        inputs = self.dense(last_hidden_state)
        inputs = torch.split(inputs, self.inner_dim * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.inner_dim], inputs[..., self.inner_dim:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = self.sinusoidal_position_embedding(self.inner_dim, 'zero', inputs)
            cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
            sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        # logits = add_mask_tril(logits, attention_mask)
        # pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        # logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12
        logits = add_mask_tril(logits, attention_mask)
        # scale返回
        return logits / self.inner_dim ** 0.5


model = GlobalPointer(model_path=DOWNLOADED_MODEL_PATH, ent_type_size=len(output_labels), inner_dim=64)
if LOAD_MODEL_FROM is None:
    optimizer = torch.optim.AdamW([{'params': model.model.parameters(), 'lr': config['learning_rate_for_bert'],
                                    "weight_decay": config['weight_decay']},
                                   {'params': model.dense.parameters(), 'lr': config['learning_rate_for_others'],
                                    "weight_decay": 0.0}],
                                  eps=1e-6,

                                  )

    num_batches = len(train_dataloader) / config['train_batch_size']
    total_train_steps = int(num_batches * config['epochs'])
    warmup_steps = int(0.2 * total_train_steps)

    sched = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps,
                                                      num_training_steps=total_train_steps,
                                                      lr_end=2e-5,
                                                      power=2
                                                      )
    '''
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                len(train_dataloader) * 2,
                                                                1)
    '''
model.to(config['device'])


def get_evaluate_fpr(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))
    R = set(pred)
    T = set(true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    f1 = 0 if Y + Z == 0 else 2 * X / (Y + Z)
    precision = 0 if Y == 0 else X / Y
    recall = 0 if Z == 0 else X / Z

    return f1, precision, recall


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true)
    return loss


def train():
    tr_loss = 0
    nb_tr_steps = 0
    # tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    n = 0
    optimizer.zero_grad()
    with tqdm(total=len(train_dataloader), desc="Train") as pbar:
        for idx, batch in enumerate(train_dataloader):
            ids = batch[0].to(config['device']).long()
            mask = batch[1].to(config['device']).long()
            labels = batch[2].to(config['device']).long()
            seg_ids = batch[3].to(config['device']).long()
            tr_logits = model(ids, mask, seg_ids)
            loss = loss_fun(labels, tr_logits)

            tr_loss += loss.item()

            nb_tr_steps += 1

            # gradient clipping

            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=config['max_grad_norm']
            )

            n += 1
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sched.step()

            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            pbar.update(1)

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")


def valid():
    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            ids = batch[0].to(config['device']).long()
            mask = batch[1].to(config['device']).long()
            labels = batch[2].to(config['device']).long()
            seg_ids = batch[3].to(config['device']).long()
            logits = model(ids, mask, seg_ids)
            loss += loss_fun(labels, logits)
            f1, precession, recall = get_evaluate_fpr(logits, labels)
            total_f1 += f1
            total_precision += precession
            total_recall += recall
    avg_f1 = total_f1 / (len(train_dataloader))
    avg_precision = total_precision / (len(train_dataloader))
    avg_recall = total_recall / (len(train_dataloader))
    avg_loss = loss / len(train_dataloader)

    logger.info("******************************************")
    logger.info(
        {"train_precision": avg_precision, "train_recall": avg_recall, "train_f1": avg_f1, 'train_loss': avg_loss})

    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            ids = batch[0].to(config['device']).long()
            mask = batch[1].to(config['device']).long()
            labels = batch[2].to(config['device']).long()
            seg_ids = batch[3].to(config['device']).long()
            logits = model(ids, mask, seg_ids)
            loss += loss_fun(labels, logits)
            f1, precession, recall = get_evaluate_fpr(logits, labels)
            total_f1 += f1
            total_precision += precession
            total_recall += recall
    avg_f1 = total_f1 / (len(dev_dataloader))
    avg_precision = total_precision / (len(dev_dataloader))
    avg_recall = total_recall / (len(dev_dataloader))
    avg_loss = loss / len(dev_dataloader)
    logger.info("******************************************")
    logger.info(
        {"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1, 'avg_loss': avg_loss})
    logger.info('\n')
    return avg_f1


max_acc = 0.84
if not LOAD_MODEL_FROM:

    for epoch in range(config['epochs']):

        logger.info(f"### Training epoch: {epoch + 1}")
        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[-1]['lr']
        logger.info(f'### LR_bert = {lr1}')
        logger.info(f'### LR_Linear = {lr2}')
        train()
        result = valid()
        if result >= max_acc:
            max_acc = result
            torch.save(model.state_dict(),
                       f'/home/zhr/JDNER/outputs/{MODEL_NAME}_v{VER}_{max_acc}.pt')
        gc.collect()

else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/uer_large_v1_3_0.8053813472437676.pt'))
    print('Model loaded.')
    # valid()

##################################################################################
# test_set = dataset(torch.load('./datasets/JDNER/test'))
# test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)
# 读取测试集文本
model = GlobalPointer(model_path=DOWNLOADED_MODEL_PATH, ent_type_size=len(output_labels), inner_dim=64)
model.load_state_dict(torch.load(f'./outputs/{MODEL_NAME}_v{VER}_{max_acc}.pt'))


class GlobalPointerNERPredictor(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = labels_to_ids
        self.tokenizer = tokernizer
        self.device = config['device']

        self.id2cat = ids_to_labels

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = get_token_mapping(text, tokens)

        input_ids = self.tokenizer(text)
        input_ids, input_mask, segment_ids = np.asarray(input_ids['input_ids'], dtype='int32'), np.asarray(
            input_ids['attention_mask'], dtype='int32'), np.asarray(input_ids['token_type_ids'], dtype='int32')

        zero = [0 for i in range(config['max_length'])]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), config['max_length'])])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
        }
        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        return self._convert_to_transfomer_ids(text)

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=0
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()
        model.eval()
        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = self.module(**inputs)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []

        for category, start, end in zip(*np.where(scores > threshold)):
            if end - 1 > token_mapping[-1][-1]:
                break
            if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start - 1][0],
                    "end_idx": token_mapping[end - 1][-1],
                    "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                    "type": self.id2cat[category]
                }

                if entitie_['entity'] == '':
                    continue

                entities.append(entitie_)

        return entities


ner_predictor_instance = GlobalPointerNERPredictor(model, tokenizer)

predict_results = []

with open('./preliminary_test_a/sample_per_line_preliminary_A.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for _line in tqdm(lines):
        label = len(_line) * ['O']
        for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
            if 'I' in label[_preditc['start_idx']]:
                continue
            if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                continue
            if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                continue

            label[_preditc['start_idx']] = 'B-' + _preditc['type']
            label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
                'start_idx']) * [('I-' + _preditc['type'])]

        predict_results.append([_line, label])

with open('gobal_pointer_baseline.txt', 'w', encoding='utf-8') as f:
    for _result in predict_results:
        for word, tag in zip(_result[0], _result[1]):
            if word == '\n':
                continue
            f.write(f'{word} {tag}\n')
        f.write('\n')