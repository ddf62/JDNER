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
from transformers import AutoTokenizer, AutoModel, get_polynomial_decay_schedule_with_warmup, AutoConfig
from torch import cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu

start_time = time.localtime(time.time())
start_time_str = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'
# VERSION FOR SAVING MODEL WEIGHTS
VER = 1

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = './outputs/'
LOAD_DATA_FROM ='yes'
# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = './prev_trained_model/bert-base-chinese'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'bert_base_chinese'
OUTPUT_EVAL_FILE = os.path.join('./outputs/logs', 'eval_results_' + start_time_str + '.txt')
logfile = './outputs/logs/' + MODEL_NAME + '_log_' + start_time_str + '.log'
log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                               datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

#################################################################
# config
config = {'model_name': MODEL_NAME,
          'max_length': 256,
          'train_batch_size': 32,
          'valid_batch_size': 128,
          'epochs': 1,
          'learning_rate_for_bert': 5e-5,
          'learning_rate_for_others': 3e-3,
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True

###############################################################
# 转化为ner标签
# LOAD_TOKENS_FROM 为存储ner标签csv的路径，处理一次，以后就不用处理了
# Ner标签采用BIO标记法
# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', '1', '2', '3', '4', '5', '6',
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


def read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": get_entity(labels)})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": get_entity(labels)})
    return lines


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
        text = data['words']
        word_labels = data['labels']
        # TOKENIZE TEXT：生成input_ids, input_mask
        text_t = [cls] + text + [sep] + [pad] * (config['max_length'] - len(text) - 2)
        '''
        labels = np.zeros((len(output_labels), config['max_length'], config['max_length']))
        for ent, start, end in word_labels:
            labels[labels_to_ids[ent], start, end] = 1
        '''
        input_ids = tokenizer.convert_tokens_to_ids(text_t)
        input_mask = [1] * (len(text) + 2) + [0] * (config['max_length'] - len(text) - 2)
        # CONVERT TO TORCH TENSORS
        df[index] = {'input_ids': input_ids,
                     'attention_mask': input_mask,
                     'labels': word_labels, 'text': text_t}
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
        output = {}
        set = self.data[index]
        output["input_ids"] = torch.tensor(set["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(set["attention_mask"], dtype=torch.long)
        labels = np.zeros((len(output_labels), config['max_length'], config['max_length']))
        for ent, start, end in set['labels']:
            labels[labels_to_ids[ent], start, end] = 1
        output["labels"] = torch.tensor(labels.tolist(), dtype=torch.long)
        output['text'] = set['text']

        return output

    def __len__(self):
        return self.len

class Collate:
    def __init__(self):
        pass

    def __call__(self, batch):
        output = dict()
        output['input_ids'] = torch.stack([sample['input_ids'] for sample in batch])
        output['attention_mask'] = torch.stack([sample['attention_mask'] for sample in batch])
        output['labels'] = torch.stack([sample['labels'] for sample in batch])
        output['text'] = [sample['text'] for sample in batch]
        return output
#################################################################


if LOAD_DATA_FROM is None:
    # 读取训练文件

    print('preprocess train')
    train_text = read_text('./datasets/JDNER/train.txt')
    train_df = preprocessor(train_text, tokenizer)
    # train_set = dataset(train_df)
    torch.save(train_df, '/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/train')
    print("TRAIN Dataset: {}".format(len(train_df)))

    # 读取开发集文本
    print('preprocess dev')
    dev_text = read_text('./datasets/JDNER/dev.txt')
    dev_df = preprocessor(dev_text, tokenizer)
    # dev_set = dataset(dev_df)
    torch.save(dev_df, '/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/dev')
    print("DEV Dataset: {}".format(len(dev_df)))

    # 读取测试集文本
    print('preprocess test')
    test_text = read_test_text('./datasets/JDNER/test.txt')
    test_df = preprocessor(test_text, tokenizer)
    # test_set = dataset(test_df)
    torch.save(test_df, '/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/test')
    print("TEST Dataset: {}".format(len(test_df)))

train_set = dataset(torch.load('/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/train'))
dev_set = dataset(torch.load('/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/dev'))
test_set = dataset(torch.load('/home/yqy/jdner-newbaseline/JDNER/datasets/JDNER/test'))
#train_set = dataset(train_df)
#dev_set = dataset(dev_df)
#test_set = dataset(test_df)
# 生成dataloader
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 8,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 12,
               'pin_memory': True
               }
collate = Collate()
train_dataloader = DataLoader(train_set, **train_params, collate_fn=collate)
dev_dataloader = DataLoader(dev_set, **test_params, collate_fn=collate)
test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)


#########################################################################
# 模型

class GlobalPointer(nn.Module):
    def __init__(self, model_path, ent_type_size, inner_dim, RoPE=True, hidden_size=768):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.mdoel = AutoModel.from_config(self.config)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        # [seq_len,1]
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        # [output_dim/2]
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        # [seq_len, output_dim/2]
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # [batch_size, seq_length, output_dim/2, 2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(config['device'])
        return embeddings

    def forward(self, input_ids, attention_mask):
        context_outputs = self.mdoel(input_ids, attention_mask)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


model = GlobalPointer(model_path=DOWNLOADED_MODEL_PATH, ent_type_size=len(output_labels), inner_dim=64)
optimizer = torch.optim.Adam([{'params': model.mdoel.parameters(), 'lr': config['learning_rate_for_bert']},
                              {'params': model.dense.parameters()}], lr=config['learning_rate_for_others'])
num_batches = len(train_dataloader) / config['train_batch_size']
total_train_steps = int(num_batches * config['epochs'])
warmup_steps = int(0.2 * total_train_steps)
sched = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=warmup_steps,
                                                  num_training_steps=total_train_steps,
                                                  lr_end=2e-5,
                                                  power=2
                                                  )

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
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    pred = []
    label = []
    n = 0
    with tqdm(total=len(train_dataloader), desc="Train") as pbar:
        for idx, batch in enumerate(train_dataloader):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            labels = batch['labels'].to(config['device'], dtype=torch.long)

            tr_logits = model(ids, mask)

            loss = loss_fun(labels, tr_logits)

            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

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

            pbar.set_postfix({'loss': '{0:1.5f}'.format(tr_loss / nb_tr_steps)})
            pbar.update(1)

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")


def valid(prefix=""):
    model.eval()
    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            labels = batch['labels'].to(config['device'], dtype=torch.long)
            logits = model(ids, mask)
            loss += loss_fun(labels, logits)
            f1, precession, recall = get_evaluate_fpr(logits, labels)
            total_f1 += f1
            total_precision += precession
            total_recall += total_recall
    avg_f1 = total_f1 / (len(dev_dataloader))
    avg_precision = total_precision / (len(dev_dataloader))
    avg_recall = total_recall / (len(dev_dataloader))
    avg_loss = loss / len(dev_dataloader)
    print("******************************************")
    logger.info({"train_precision": avg_precision, "train_recall": avg_recall, "train_f1": avg_f1, 'train_loss': avg_loss})
    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            labels = batch['labels'].to(config['device'], dtype=torch.long)
            logits = model(ids, mask)
            loss += loss_fun(labels, logits)
            f1, precession, recall = get_evaluate_fpr(logits, labels)
            total_f1 += f1
            total_precision += precession
            total_recall += total_recall
    avg_f1 = total_f1 / (len(dev_dataloader))
    avg_precision = total_precision / (len(dev_dataloader))
    avg_recall = total_recall / (len(dev_dataloader))
    avg_loss = loss / len(dev_dataloader)
    print("******************************************")
    logger.info({"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1, 'avg_loss': avg_loss})
    return avg_f1


if not LOAD_MODEL_FROM:
    max_acc = 0
    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[-1]['lr']
        print(f'### LR_bert = {lr1}\n### LR_Linear = {lr2}\n')
        train()
        torch.save(model.state_dict(),
                   f'/home/yqy/jdner-newbaseline/JDNER/outputs/{MODEL_NAME}_v{VER}_temporary_{epoch}.pt')
        result = valid()
        torch.cuda.empty_cache()
        # gc释放内存
        gc.collect()
        if result >= max_acc:
            max_acc = result
            torch.save(model.state_dict(), f'/home/yqy/jdner-newbaseline/JDNER/outputs/{MODEL_NAME}_v{VER}_{max_acc}.pt')

else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/bert_base_chinese_v1_temporary_0.pt'))
    print('Model loaded.')
    #valid()


##################################################################################
# 推理
def decode_ent(text, pred_matrix, threshold=0):
    # print(text)
    ent_list = {}
    if tokenizer.pad_token in text:
        length = text.index(tokenizer.pad_token)
    else:
        lenght = len(text)
    for ent_type_id, token_start_index, toekn_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = ids_to_labels[ent_type_id]
        ent_char_span = [token_start_index, toekn_end_index]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]
        if ent_char_span[0] >= length:
            continue
        if ent_char_span[0] == 0:
            ent_char_span[0] = 1
        if ent_char_span[1] >= length:
            ent_char_span[1] = length- 1


        ent_type_dict = ent_list.get(ent_type, [])
        ent_type_dict.append(ent_char_span)
        ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)
    return ent_list


def predict():
    fp = open('test_prediction.txt', 'w')
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            text = batch['text']
            tr_logits = model(ids, mask)
            preds = tr_logits.detach().cpu().numpy()
            for i in range(len(preds)):
                labels = decode_ent(text[i], preds[i])
                index = 0
                for key in labels.keys():
                    for span in labels[key]:
                        for index in range(span[0], span[1] + 1):
                            text[i][index] = text[i][index] + ' ' + 'I-' + key + '\n'
                        text[i][span[0]] = text[i][span[0]] + ' ' + 'B-' + key + '\n'
                for jk in range(len(text[i])):
                    fp.write(text[i][jk])
                fp.write('\n')
    fp.close()


predict()
