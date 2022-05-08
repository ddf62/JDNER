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
from torch import kl_div, nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertTokenizer, \
    get_polynomial_decay_schedule_with_warmup, AutoConfig
from torch import cuda
from model.configuration_nezha import NeZhaConfig
from tools.utils import FGM, ConditionalLayerNorm
from model.modeling_nezha import NeZhaModel
from model.configuration_nezha import NeZhaConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0,1,2,3 for four gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
DOWNLOADED_MODEL_PATH = './prev_trained_model/nezha-mlm-0.4'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'nazha-mlm-0.4-token-block-fgm-dropout-wei-all'
log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                               datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger()

if LOAD_MODEL_FROM is None:
    logfile = './outputs/logs/' + MODEL_NAME + '_log_' + start_time_str + '.log'

    logging.basicConfig(level=logging.DEBUG)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.info(MODEL_NAME)

torch.set_printoptions(threshold=np.inf)
logger.info('修改了梯度裁剪位置')
#################################################################
# config
config = {'model_name': MODEL_NAME,
          'max_length': 256,
          'train_batch_size': 32,
          'valid_batch_size': 64,
          'epochs': 8,
          'learning_rate_for_bert': 5e-5,
          'learning_rate_for_others': 2e-3,
          'weight_decay': 1e-6,
          'epsilon': 0.5,
          'max_grad_norm': 10,
          'norm': 1,
          'T_max': 2,
          'thread': 0.82,
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
                if splits[0] == '':
                    words.append(' ')
                else:
                    words.append(splits[0].lower())
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
        # text = data['words']
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


tokenizer = BertTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH, do_lower_case=True)
tokenizer.add_tokens(' ')
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

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 6,
               'pin_memory': True
               }
collate = Collate(tokenizer)

if LOAD_DATA_FROM is None:
    # 读取训练文件

    print('preprocess train')
    train_text = read_text('./datasets/JDNER/train_wei.txt')
    train_df = preprocessor(train_text, tokenizer)
    train_set = dataset(train_df)
    print("TRAIN Dataset: {}".format(len(train_df)))

    # 读取开发集文本
    print('preprocess dev')
    dev_text = read_text('./datasets/JDNER/dev.txt')
    dev_df = preprocessor(dev_text, tokenizer)
    dev_set = dataset(dev_df)
    print("DEV Dataset: {}".format(len(dev_df)))

    train_dataloader = DataLoader(train_set, **train_params, collate_fn=collate)
    dev_dataloader = DataLoader(dev_set, **test_params, collate_fn=collate)


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
        configs = NeZhaConfig.from_pretrained(model_path)
        self.model = NeZhaModel.from_pretrained(model_path, config=configs)
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

    def forward(self, input_ids, attention_mask, seg):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=seg,
            # return_dict=True,
            # output_hidden_states=True
        )  # .hidden_states

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


model = GlobalPointer(model_path=DOWNLOADED_MODEL_PATH, ent_type_size=len(output_labels), inner_dim=64, hidden_size=768)
# model.model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.AdamW([{'params': model.model.parameters(), 'lr': config['learning_rate_for_bert'],
                                "weight_decay": config['weight_decay']},
                               {'params': model.dense.parameters(), 'lr': config['learning_rate_for_others'],
                                "weight_decay": 0.0}],
                              eps=1e-6,
                              )
'''
sched = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=warmup_steps,
                                                  num_training_steps=total_train_steps,
                                                  lr_end=2e-5,
                                                  power=2
                                                  )
'''
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             len(train_dataloader) * config['norm'],
                                                             config['T_max'])

fgm = FGM(model)
model.to(config['device'])


# kl_div = nn.KLDivLoss(reduction='batchmean')
# model.load_state_dict(torch.load('outputs/uer_large_token_v1_0.8117350654732964.pt'))

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


global_step = 0


def train():
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    n = 0
    global global_step
    with tqdm(total=len(train_dataloader), desc="Train") as pbar:
        for idx, batch in enumerate(train_dataloader):
            ids = batch[0].to(config['device'], dtype=torch.long)
            mask = batch[1].to(config['device'], dtype=torch.long)
            labels = batch[2].to(config['device'], dtype=torch.long)
            seg = batch[3].to(config['device'], dtype=torch.long)

            tr_logits = model(ids, mask, seg)
            loss = loss_fun(labels, tr_logits)
            tr_loss += loss.item()
            # r-drop
            # tr_logits2 = model(ids, mask, seg)
            # loss1 = loss_fun(labels, tr_logits)
            # loss2 = loss_fun(labels, tr_logits2)
            # print(kl_div(tr_logits.log(), tr_logits2), kl_div(tr_logits2.log(), tr_logits))
            # kl_loss = (kl_div(tr_logits, tr_logits2).sum(-1) + kl_div(tr_logits2, tr_logits).sum(-1)) / 2
            # tr_loss += (loss1.item() + loss2.item())/2
            # loss = (loss1 + loss2) / 2 + kl_loss

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            # gradient clipping

            n += 1
            # backward pass

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=config['max_grad_norm']
            )

            # fgm.attack(epsilon=config['epsilon'], emb_name='bert.embeddings.word_embeddings.weight')
            fgm.attack(epsilon=config['epsilon'], emb_name='model.embeddings.word_embeddings.weight')
            logits_adv = model(ids, mask, seg)
            loss_adv = loss_fun(labels, logits_adv)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=config['max_grad_norm']
            )

            # fgm.restore(emb_name='bert.embeddings.word_embeddings.weight')
            fgm.restore(emb_name='model.embeddings.word_embeddings.weight')
            optimizer.step()
            sched.step()
            optimizer.zero_grad()
            pbar.set_postfix({'loss': '{0:1.5f}'.format(tr_loss / nb_tr_steps)})
            pbar.update(1)
            global_step += 1
            if global_step % 200 == 0:
                logger.info(f'steps: {global_step}')
                f1 = valid_dev()
                if f1 > config['thread']:
                    torch.save(model.state_dict(),
                               f'/home/zbg/Reinforcement/强化学习算法2021/JDNER/outputs/{MODEL_NAME}_{global_step}_{f1}.pt')
                model.train()

    epoch_loss = tr_loss / nb_tr_steps
    logger.info(f"Training loss epoch: {epoch_loss}")


def valid(prefix=""):
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            ids = batch[0].to(config['device'], dtype=torch.long)
            mask = batch[1].to(config['device'], dtype=torch.long)
            labels = batch[2].to(config['device'], dtype=torch.long)
            seg = batch[3].to(config['device'], dtype=torch.long)

            logits = model(ids, mask, seg)
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
            ids = batch[0].to(config['device'], dtype=torch.long)
            mask = batch[1].to(config['device'], dtype=torch.long)
            labels = batch[2].to(config['device'], dtype=torch.long)
            seg = batch[3].to(config['device'], dtype=torch.long)

            logits = model(ids, mask, seg)
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


def valid_dev():
    model.eval()
    total_f1, total_precision, total_recall = 0., 0., 0.
    loss = 0
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            ids = batch[0].to(config['device'], dtype=torch.long)
            mask = batch[1].to(config['device'], dtype=torch.long)
            labels = batch[2].to(config['device'], dtype=torch.long)
            seg = batch[3].to(config['device'], dtype=torch.long)

            logits = model(ids, mask, seg)
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


if not LOAD_MODEL_FROM:
    max_acc = 0.82
    for epoch in range(config['epochs']):

        logger.info(f"### Training epoch: {epoch + 1}")
        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[-1]['lr']
        logger.info(f'### LR_bert = {lr1}\n### LR_Linear = {lr2}')
        train()
        result = valid()
        if result >= max_acc:
            max_acc = result
            torch.save(model.state_dict(),
                       f'/home/zbg/Reinforcement/强化学习算法2021/JDNER/outputs/{MODEL_NAME}_v{VER}_{max_acc}.pt')
        gc.collect()

##################################################################################
