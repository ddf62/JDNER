import copy
import gc
from html import entities
import logging
import nturl2path
import os

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"  # 0,1,2,3 for four gpu
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
DOWNLOADED_MODEL_CLASS = './prev_trained_model/bert-base-chinese'
DOWNLOADED_MODEL_GLOBAL = './prev_trained_model/uer-large'

MODEL_NAME = 'merge_classify_global'
# config
config = {'model_name': MODEL_NAME,
          'device': 'cuda:0' if cuda.is_available() else 'cpu',
          'max_length': 160}
torch.set_printoptions(threshold=np.inf)
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


tokenizer = BertTokenizerFast.from_pretrained(DOWNLOADED_MODEL_GLOBAL, do_lower_case=True)
'''
for i in output_labels:
    tokenizer.add_tokens('[' + i + ']')
'''

#########################################################################
class EntityClassification(nn.Module):
    def __init__(self, model_path, ent_type_size, hidden_size=768):
        super().__init__()
        # self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.ent_type_size = ent_type_size
        self.hidden_size = hidden_size
        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, ent_type_size)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-1][:, 0, :]
        inputs = self.dense(last_hidden_state)
        return inputs


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
    def __init__(self, model_path, ent_type_size, inner_dim=64, RoPE=True, hidden_size=1024):
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


class MergeClassGlobal(nn.Module):
    def __init__(self, class_path, class_pram, global_path, ent_size, tokenizer, freeze):
        super().__init__()
        self.classify = EntityClassification(class_path, ent_size)
        # self.classify.load_state_dict(torch.load(class_pram))
        if freeze:
            for p in self.classify.parameters():
                p.requires_grad = False
        self.ent_size = ent_size
        self.globalpointer = GlobalPointer(global_path, 1)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids, length):
        class_ = self.classify(input_ids, attention_mask, token_type_ids)
        true_ent = (class_ > 0).nonzero()
        inputs_ = torch.zeros((len(true_ent), input_ids.shape[1]))
        attention_mask_ = torch.zeros((len(true_ent), input_ids.shape[1]))
        token_type_ids_ = torch.zeros((len(true_ent), input_ids.shape[1]))
        for idx, en in enumerate(true_ent):
            b, e = en[0].item(), en[1].item()
            inputs_[idx] = input_ids[b].detach().cpu()
            attention_mask_[idx] = attention_mask[b].detach().cpu()
            token_type_ids_[idx] = token_type_ids[b].detach().cpu()

            inputs_[idx][length[b][0] - 1], inputs_[idx][length[b][0]] = self.tokenizer.convert_tokens_to_ids(
                '[' + ids_to_labels[e] + ']'), self.tokenizer.sep_token_id
            attention_mask_[idx][length[b][0]] = 1
            token_type_ids_[idx][length[b][0] - 1:] = 1

        inputs_ = inputs_.to(config['device']).long()
        attention_mask_ = attention_mask_.to(config['device']).long()
        token_type_ids_ = token_type_ids_.to(config['device']).long()

        outputs = self.globalpointer(inputs_, attention_mask_, token_type_ids_)
        return outputs, true_ent

'''
model = MergeClassGlobal(class_path=DOWNLOADED_MODEL_CLASS,
                         class_pram='outputs/bert-base-chinese_classify_v1_racall_0.9912525035931105.pt',
                         global_path=DOWNLOADED_MODEL_GLOBAL, ent_size=len(output_labels),
                         tokenizer=tokenizer, freeze=True)
model.globalpointer.model.resize_token_embeddings(len(tokenizer))
'''
model = GlobalPointer(model_path=DOWNLOADED_MODEL_GLOBAL, ent_type_size=len(output_labels))
##################################################################################
# test_set = dataset(torch.load('./datasets/JDNER/test'))
# test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)
# 读取测试集文本
model.to(config['device'])
model.load_state_dict(torch.load(f'/home/zhr/JDNER/outputs/uer_large_fgm_v1_0.8100679180006238.pt'))


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

with open('./preliminary_test_a/preliminary_test_a/sample_per_line_preliminary_A.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    num = 0
    for _line in tqdm(lines):
        label = len(_line) * ['O']
        if num < 100000:
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
        num += 1
        predict_results.append([_line, label])

with open('gobal_pointer_baseline_0.801.txt', 'w', encoding='utf-8') as f:
    for _result in predict_results:
        for word, tag in zip(_result[0], _result[1]):
            if word == '\n':
                continue
            f.write(f'{word} {tag}\n')
        f.write('\n')