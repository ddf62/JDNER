import copy
import gc
from html import entities
import logging
import nturl2path
import os
from sklearn.model_selection import train_test_split
# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
from collections import Counter
import time
from tracemalloc import start
from turtle import forward
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
from transformers import logging

from model.modeling_nezha import NeZhaModel

logging.set_verbosity_warning()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu
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
DOWNLOADED_MODE=['bert-base-chinese', './prev_trained_model/uer-large', 'hfl/chinese-roberta-wwm-ext', './prev_trained_model/nezha-mlm-0.4']
hidden_size_=[768,1024,768, 768]
load_from_=[
            f'outputs/bert-base-7epoch_v1_0.8553572532022676.pt',
            f'outputs/uer-large-base-7epoch_v1_0.9082318977332601.pt',
            f'outputs/roberta-base-7epoch_v1_0.8491710035719371.pt',
            f'outputs/nazha-mlm-0.4-base-7epoch_v1_0.8617323533051509.pt'
]

# config
config = {'device': 'cuda:0' if cuda.is_available() else 'cpu',
          'max_length': 160}
#################################################################
# config

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
test_params = {'batch_size': 1,
               'shuffle': False,
               'num_workers': 6,
               'pin_memory': True
               }
tokenizer = BertTokenizerFast.from_pretrained(DOWNLOADED_MODE[-1], do_lower_case=True)
tokenizer.add_tokens(' ')
collate = Collate(tokenizer)

#########################################################################

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
    def __init__(self, model_path, ent_type_size, inner_dim=64, hidden_size=1024):
        super().__init__()
        #configs = NeZhaConfig.from_pretrained(model_path, output_hidden_states=True)
        #self.model = NeZhaModel.from_pretrained(model_path, config=configs)
        self.model = AutoModel.from_pretrained(model_path)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        # self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        )

        # self.type_embedding = nn.Embedding(len(output_labels), 256)
        # self.condition = ConditionalLayerNorm(1024, 256, eps=1e-12)

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

    def forward(self, input_ids, attention_mask, seg, train_=False):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=seg,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states
        if train_==True:
            last_hidden_state = torch.cat([outputs[i] for i in [-1,-2,-3,4]], dim=0)
            attention_mask = torch.cat([attention_mask for i in range(4)], dim=0)
        else:
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

class GlobalPointer_N(nn.Module):
    def __init__(self, model_path, ent_type_size, inner_dim=64, hidden_size=1024):
        super().__init__()
        #configs = NeZhaConfig.from_pretrained(model_path, output_hidden_states=True)
        self.model = NeZhaModel.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        # self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        )

        # self.type_embedding = nn.Embedding(len(output_labels), 256)
        # self.condition = ConditionalLayerNorm(1024, 256, eps=1e-12)

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

    def forward(self, input_ids, attention_mask, seg, train_=False):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=seg,
            # return_dict=True,
            # output_hidden_states=True
        ) #.hidden_states
        if train_==True:
            last_hidden_state = torch.cat([outputs[i] for i in [-1,-2,-3,4]], dim=0)
            attention_mask = torch.cat([attention_mask for i in range(4)], dim=0)
        else:
            last_hidden_state = outputs[0]
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

#####################################################
data_text=[]
with open('datasets/JDNER/unlabeled_train_data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data_text.append({'words': list(line[:-1]), 'labels': get_entity(['O'] * len(list(line)))})
train_text, test_text = train_test_split(data_text, test_size=10000, random_state=52)
print(test_text[0])
test_df = preprocessor(copy.deepcopy(test_text), tokenizer)
test_set = dataset(test_df)
test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)
# train_text, test_text = train_test_split(data_text, test_size=10000, random_state=42)
# print(test_text[9997:],len(test_text),'******************************')


# 推理
def decode_ent(text, pred_matrix, threshold=0):
    # print(text)
    ent_list = {}
    length = len(text)
    for ent_type_id, token_start_index, toekn_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = ids_to_labels[ent_type_id]
        ent_char_span = [token_start_index - 1, toekn_end_index - 1, pred_matrix[ent_type_id][token_start_index][toekn_end_index]]

        ent_type_dict = ent_list.get(ent_type, [])
        ent_type_dict.append(ent_char_span)
        ent_list.update({ent_type: ent_type_dict})
    return ent_list


score = []
for i in range(4):
    if 'nezha' in DOWNLOADED_MODE[i]:
        model = GlobalPointer_N(model_path=DOWNLOADED_MODE[i], ent_type_size=len(output_labels),
                          hidden_size=hidden_size_[i])
        
    else:
        model = GlobalPointer(model_path=DOWNLOADED_MODE[i], ent_type_size=len(output_labels),
                          hidden_size=hidden_size_[i])
        model.model.resize_token_embeddings(len(tokenizer))
    model.to(config['device'])
    model.load_state_dict(torch.load(load_from_[i]))
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
            if i == 0:
                score.append(preds/len(DOWNLOADED_MODE))
            else:
                score[tmp] += preds/len(DOWNLOADED_MODE)
            tmp += 1

fp = open('datasets/JDNER/presudo_7epoch-0520.txt', 'w')
for i in range(len(test_text)):
    preds = score[i]
    preds[:, [0, -1]] -= np.inf
    preds[:, :, [0, -1]] -= np.inf
    
    labels_ = ['O'] * len(test_text[i]['words'])
    scores = [0] * len(test_text[i]['words'])
    starts = [0] * len(test_text[i]['words'])
    ends = [0] * len(test_text[i]['words'])
    labels = decode_ent(test_text[i]['words'], preds, threshold=0)
    index = 0
    for key in labels.keys():
        for span in labels[key]:
            max_s = max(scores[span[0]: span[1] + 1])
            if max_s < span[2]:
                head_0, tail_0 = starts[span[0]], ends[span[0]]
                head_1, tail_1 = starts[span[1]], ends[span[1]]
                if head_0 != 0 and tail_0 != 0:
                    labels_[head_0: tail_0 + 1] = ['O'] * (tail_0+ 1 - head_0)
                    scores[head_0: tail_0 + 1] = [0] * (tail_0 + 1 -head_0)
                    starts[head_0: tail_0 + 1] =  [0] * (tail_0 + 1 -head_0)
                    ends[head_0: tail_0 + 1] =  [0] * (tail_0 + 1 -head_0)
                if head_1 != 0 and tail_1 !=0:
                    labels_[head_1: tail_1 + 1] = ['O'] * (tail_1+ 1 - head_1)
                    scores[head_1: tail_1 + 1] = [0] * (tail_1 + 1 -head_1)
                    starts[head_1: tail_1 + 1] =  [0] * (tail_1 + 1 -head_1)
                    ends[head_1: tail_1 + 1] =  [0] * (tail_1 + 1 -head_1)
                labels_[span[0]] = 'B-' + key
                labels_[span[0] + 1: span[1] + 1] = (span[1] - span[0]) * [('I-' + key)]
                scores[span[0]: span[1] + 1] = [span[2]] * (span[1] - span[0] + 1)
                starts[span[0]: span[1] + 1] = [span[0]] * (span[1] - span[0] + 1)
                ends[span[0]: span[1] + 1] = [span[1]] * (span[1] - span[0] + 1)
    for jk in range(len(test_text[i]['words'])):
        if len(test_text[i]['words'][jk].split(' ')) == 1 or test_text[i]['words'][jk] == ' ':
            fp.write(test_text[i]['words'][jk] + ' ' + labels_[jk] + '\n')
        else:
            fp.write(test_text[i]['words'][jk] + ' ' + labels_[jk] + '\n')
    fp.write('\n')
fp.close()