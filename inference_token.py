import copy
import gc
import logging
import os

# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
from collections import Counter
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, get_polynomial_decay_schedule_with_warmup, \
    AutoConfig
from torch import cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 0,1,2,3 for four gpu

start_time = time.localtime(time.time())
start_time_str = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'
# VERSION FOR SAVING MODEL WEIGHTS
VER = 1

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = None  # './outputs/'
LOAD_DATA_FROM = None
# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = './prev_trained_model/uer-large'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'uer_large_token'
log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                               datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger()
if LOAD_MODEL_FROM is None:
    OUTPUT_EVAL_FILE = os.path.join('./outputs/logs', 'eval_results_' + start_time_str + '.txt')
    logfile = './outputs/logs/' + MODEL_NAME + '_log_' + start_time_str + '.log'

    logging.basicConfig(level=logging.DEBUG)
    logger.info(MODEL_NAME)

torch.set_printoptions(threshold=np.inf)

#################################################################
# config
config = {'model_name': MODEL_NAME,
          'max_length': 256,
          'train_batch_size': 64,
          'valid_batch_size': 128,
          'epochs': 20,
          'learning_rate_for_bert': 5e-5,
          'learning_rate_for_others': 5e-4,
          'weight_decay': 1e-6,
          'max_grad_norm': 4,
          'device': 'cuda' if cuda.is_available() else 'cpu'}
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


def read_test_text(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.read().split('\n'):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": get_entity(labels)})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                if splits[0] == '':
                    words.append(' ')
                else:
                    words.append(splits[0])
                if len(splits) > 1 and line != ' ':
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    if line == ' ':
                        words[-1] = ' '
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": get_entity(labels)})
    return lines


def preprocessor(df, tokenizer):
    pad = tokenizer.pad_token
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    # GET TEXT AND WORD LABELS
    for index in tqdm(range(len(df))):
        data = df[index]
        text = [i.lower() for i in data['words']]
        word_labels = data['labels']
        # TOKENIZE TEXT：生成input_ids, input_mask
        text_t = [cls] + text + [sep]
        '''
        labels = np.zeros((len(output_labels), config['max_length'], config['max_length']))
        for ent, start, end in word_labels:
            labels[labels_to_ids[ent], start, end] = 1
        '''
        input_ids = tokenizer.convert_tokens_to_ids(text_t)
        input_mask = [1] * (len(text) + 2)
        # CONVERT TO TORCH TENSORS
        df[index] = {'input_ids': input_ids,
                     'attention_mask': input_mask,
                     'labels': word_labels}
    return df


tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH, do_lower_case=True)

#########################################################################
# 定义dataset

LABEL_ALL_SUBTOKENS = True


class dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        # GET TEXT AND WORD LABEL
        return self.data[index]

    def __len__(self):
        return self.len


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        maxlen = max([len(idx) for idx in input_ids])
        labels = np.zeros((len(batch), len(output_labels), maxlen, maxlen))
        for sample in range(len(batch)):
            for ent, start, end in batch[sample]['labels']:
                labels[sample, labels_to_ids[ent], start + 1, end + 1] = 1
        # 补全
        input_ids = [s + [self.tokenizer.pad_token_id] * (maxlen - len(s)) for s in input_ids]
        attention_mask = [s + [0] * (maxlen - len(s)) for s in attention_mask]

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        seg_id = torch.zeros((len(batch), maxlen))
        return input_ids, attention_mask, labels, seg_id


#################################################################


# train_set = dataset(train_df)
# dev_set = dataset(dev_df)
# test_set = dataset(test_df)
# 生成dataloader
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 6,
                'pin_memory': True
                }

test_params = {'batch_size': 1,
               'shuffle': False,
               'num_workers': 6,
               'pin_memory': True
               }
collate = Collate(tokenizer)


#########################################################################
# 模型
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
    def __init__(self, model_path, ent_type_size, inner_dim, hidden_size=1024):
        super().__init__()

        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        '''
        nn.Sequential(
            nn.Dropout(p=0.22),
            nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        )'''

        self.RoPE = True

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

    def forward(self, input_ids, attention_mask, seg):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=seg,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        last_hidden_state = outputs[-1]
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

model.to(config['device'])
model.load_state_dict(torch.load('outputs/uer_large_pretrained_token_v1_0.8127603412616385.pt'))
##################################################################################
test_text = read_test_text('./datasets/JDNER/test.txt')
test_df = preprocessor(test_text, tokenizer)
test_set = dataset(test_df)
test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)


# 推理
def decode_ent(text, pred_matrix, threshold=0):
    # print(text)
    ent_list = {}
    length = len(text)
    for ent_type_id, token_start_index, toekn_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = ids_to_labels[ent_type_id]
        ent_char_span = [token_start_index - 1, toekn_end_index - 1]

        ent_type_dict = ent_list.get(ent_type, [])
        ent_type_dict.append(ent_char_span)
        ent_list.update({ent_type: ent_type_dict})
    return ent_list


def predict():
    test_text = read_test_text('./datasets/JDNER/test.txt')
    fp = open('test_prediction.txt', 'w')
    tmp = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            ids = batch[0].to(config['device'], dtype=torch.long)
            mask = batch[1].to(config['device'], dtype=torch.long)
            seg = batch[3].to(config['device'], dtype=torch.long)

            tr_logits = model(ids, mask, seg)[0]
            preds = tr_logits.detach().cpu().numpy()
            preds[:, [0, -1]] -= np.inf
            preds[:, :, [0, -1]] -= np.inf

            labels_ = ['O'] * len(test_text[tmp]['words'])
            labels = decode_ent(test_text[tmp]['words'], preds)
            index = 0
            for key in labels.keys():
                for span in labels[key]:

                    if 'I' in labels_[span[0]]:
                        continue
                    if 'B' in labels_[span[0]] and 'O' not in labels_[span[1]]:
                        continue
                    if 'O' in labels_[span[0]] and 'B' in labels_[span[1]]:
                        continue

                    labels_[span[0]] = 'B-' + key
                    labels_[span[0] + 1: span[1] + 1] = (span[1] - span[0]) * [('I-' + key)]

            for jk in range(len(test_text[tmp]['words'])):
                if len(test_text[tmp]['words'][jk].split(' ')) == 1 or test_text[tmp]['words'][jk] == ' ':
                    fp.write(test_text[tmp]['words'][jk] + ' ' + labels_[jk] + '\n')
                else:
                    fp.write(test_text[tmp]['words'][jk] + ' ' + labels_[jk] + '\n')
            fp.write('\n')
            tmp += 1
    fp.close()


predict()
