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
from transformers import AutoTokenizer, AutoModel, get_polynomial_decay_schedule_with_warmup
from torch import cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu

start_time = time.localtime(time.time())
start_time_str = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'
# VERSION FOR SAVING MODEL WEIGHTS
VER = 1

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = None
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
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.NOTSET)
logger.addHandler(file_handler)

#################################################################
# config
config = {'model_name': MODEL_NAME,
          'max_length': 256,
          'train_batch_size': 128,
          'valid_batch_size': 128,
          'epochs': 1,
          'learning_rates': 5e-5,
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True


################################################################
# 读取训练集标签
def read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
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
            lines.append({"words": words, "labels": labels})
    return lines


def read_test_text(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.read().split('\n'):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
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
            lines.append({"words": words, "labels": labels})
        print(len(lines))
    return lines


# 读取训练文件
train_text = read_text('./datasets/JDNER/train.txt')
train_df = pd.DataFrame(train_text)
# 输出前5条数据
print(train_df.head())

# 读取开发集文本
dev_text = read_text('./datasets/JDNER/dev.txt')
dev_df = pd.DataFrame(dev_text)
print(dev_df.head())

# 读取测试集文本
test_text = read_test_text('./datasets/JDNER/test.txt')
test_df = pd.DataFrame(test_text)
print(test_df.head())

###############################################################
# 转化为ner标签
# LOAD_TOKENS_FROM 为存储ner标签csv的路径，处理一次，以后就不用处理了
# Ner标签采用BIO标记法
# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', 'B-1', 'I-1', 'B-2', 'I-2', 'B-3', 'I-3', 'B-4', 'I-4', 'B-5', 'I-5', 'B-6', 'I-6',
                 'B-7', 'I-7', 'B-8', 'I-8', 'B-9', 'I-9', 'B-10', 'I-10', 'B-11', 'I-11', 'B-12', 'I-12', 'B-13',
                 'I-13', 'B-14', 'I-14', 'B-15', 'I-15', 'B-16', 'I-16', 'B-17', 'I-17', 'B-18', 'I-18', 'B-19', 'I-19',
                 'B-20', 'I-20', 'B-21', 'I-21', 'B-22', 'I-22', 'B-23', 'I-23', 'B-24', 'I-24', 'B-25', 'I-25', 'B-26',
                 'I-26', 'B-28', 'I-28', 'B-29', 'I-29', 'B-30', 'I-30', 'B-31', 'I-31', 'B-32', 'I-32', 'B-33', 'I-33',
                 'B-34', 'I-34', 'B-35', 'I-35', 'B-36', 'I-36', 'B-37', 'I-37', 'B-38', 'I-38', 'B-39', 'I-39', 'B-40',
                 'I-40', 'B-41', 'I-41', 'B-42', 'I-42', 'B-43', 'I-43', 'B-44', 'I-44', 'B-46', 'I-46', 'B-47', 'I-47',
                 'B-48', 'I-48', 'B-49', 'I-49', 'B-50', 'I-50', 'B-51', 'I-51', 'B-52', 'I-52', 'B-53', 'I-53', 'B-54',
                 'I-54']
# 存储标签与index之间的映射
labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

#########################################################################
# 定义dataset

LABEL_ALL_SUBTOKENS = True


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls = tokenizer.cls_token
        self.sep = tokenizer.sep_token

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.words[index]
        word_labels = self.data.labels[index]
        # 将文字label转化为数字id
        labels = [labels_to_ids[i] for i in word_labels]
        # TOKENIZE TEXT：生成input_ids, input_mask
        text = [self.cls] + text + [self.sep]
        labels = [-100] + labels + [-100]
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_mask = [1] * len(input_ids)
        # CONVERT TO TORCH TENSORS
        item = {'input_ids': input_ids, 'attention_mask': input_mask,
                'labels': labels, 'text': text}
        return item

    def __len__(self):
        return self.len


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output['input_ids'] = [sample['input_ids'] for sample in batch]
        output['attention_mask'] = [sample['attention_mask'] for sample in batch]
        output['labels'] = [sample['labels'] for sample in batch]
        batch_max = max([len(idx) for idx in output['input_ids']])
        # 添加padding
        output['input_ids'] = [s + [self.tokenizer.pad_token_id] * (batch_max - len(s)) for s in output['input_ids']]
        output['attention_mask'] = [s + [0] * (batch_max - len(s)) for s in output['attention_mask']]
        output['labels'] = [s + [-100] * (batch_max - len(s)) for s in output['labels']]

        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output['labels'] = torch.tensor(output['labels'], dtype=torch.long)
        output['text'] = [sample['text'] for sample in batch]
        return output


#################################################################
print("TRAIN Dataset: {}".format(train_df.shape))
print("DEV Dataset: {}".format(dev_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH, do_lower_case=True)
train_set = dataset(train_df, tokenizer, config['max_length'])
dev_set = dataset(dev_df, tokenizer, config['max_length'])
test_set = dataset(test_df, tokenizer, config['max_length'])
collate = Collate(tokenizer=tokenizer)
# 生成dataloader
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 2,
               'pin_memory': True
               }

train_dataloader = DataLoader(train_set, **train_params, collate_fn=collate)
dev_dataloader = DataLoader(dev_set, **test_params, collate_fn=collate)
test_dataloader = DataLoader(test_set, **test_params, collate_fn=collate)

print(len(train_dataloader))

#########################################################################
# 模型
class Mymodel(nn.Module):
    def __init__(self, model_name, num_classes, hidden_size=768, freeze=False):
        super(Mymodel, self).__init__()

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, num_classes, bias=True)
        )

    def forward(self, input_ids, attn_masks):
        outputs = self.model(input_ids, attention_mask=attn_masks)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4]
        logits = self.fc(hidden_states)
        return logits


model = Mymodel(model_name=DOWNLOADED_MODEL_PATH, num_classes=len(output_labels))
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'])
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


def get_entity(seq):
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


def entity_acc(preds, labels):
    """
        preds\labels: [num, seq_length]
    """
    print(len(preds))
    pred = []
    origin_eneit = []
    right_eneit = []
    class_info = {}
    for i in range(len(preds)):
        origin = [k for k in labels[i] if k != -100]
        pred_id = preds[i][1:len(origin)]
        pred_label = [ids_to_labels[j] for j in pred_id]
        true_label = [ids_to_labels[j] for j in origin]
        p_enit = get_entity(pred_label)
        t_enit = get_entity(true_label)
        pred.extend(p_enit)
        origin_eneit.extend(t_enit)
        right_eneit.extend([p for p in p_enit if p in t_enit])
    origin_counter = Counter([x[0] for x in origin_eneit])
    found_counter = Counter([x[0] for x in pred])
    right_counter = Counter([x[0] for x in right_eneit])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    origin = len(origin_eneit)
    found = len(pred)
    right = len(right_eneit)
    print(origin, found, right)
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return {'acc': precision, 'recall': recall, 'f1': f1}, class_info


def train():
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()
    lossn = nn.CrossEntropyLoss()
    n = 0
    pred = []
    label = []
    for idx, batch in enumerate(train_dataloader):
        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        tr_logits = model(input_ids=ids, attn_masks=mask)
        loss = 0
        pred.append(torch.argmax(tr_logits, dim=-1).detach().cpu().numpy())
        label.append(batch['labels'].detach().cpu().numpy())
        for i in range(len(tr_logits)):
            loss += lossn(tr_logits[i], labels[i])

        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 20 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

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
    train_prd = []
    train_tag = []
    for row in range(len(pred)):
        for p in range(len(pred[row])):
            train_prd.append(pred[row][p])
            train_tag.append(label[row][p])
    print(len(train_prd))
    tr_accuracy, _ = entity_acc(train_prd, train_tag)

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy['f1']}")

def valid(prefix=""):
    model.eval()
    preds = []
    labels = []
    eval_loss = 0
    eval_steps = 0
    lossn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, batch in enumerate(dev_dataloader):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            label = batch['labels'].to(config['device'], dtype=torch.long)
            tr_logits = model(input_ids=ids, attn_masks=mask)
            for i in range(len(tr_logits)):
                eval_loss += lossn(tr_logits[i], label[i])
            preds.append(tr_logits.detach().cpu().numpy())
            labels.append(batch['labels'].detach().numpy().tolist())
            eval_steps += 1
    eval_loss = eval_loss / eval_steps
    pred_label = []
    true_label = []
    for p in range(len(preds)):
        finalp = np.argmax(preds[p], axis=2).tolist()
        for pred, tlabel in zip(finalp, labels[p]):
            pred_label.append(pred)
            true_label.append(tlabel)
    eval_info, entity_info = entity_acc(pred_label, true_label)
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % ids_to_labels[int(key)])
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    with open(OUTPUT_EVAL_FILE, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))
    return results


if not LOAD_MODEL_FROM:
    max_acc = 0
    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')
        train()
        result = valid()
        torch.cuda.empty_cache()
        # gc释放内存
        gc.collect()
        if result['f1'] >= max_acc:
            max_acc = result['f1']
            torch.save(model.state_dict(), f'/home/zhr/GAIIC_track2_baseline/outputs/{MODEL_NAME}_v{VER}_{max_acc}.pt')
        torch.save(model.state_dict(),
                   f'/home/zhr/GAIIC_track2_baseline/outputs/{MODEL_NAME}_v{VER}_temporary_{epoch}.pt')
else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/longformer-large_temporary_v3_3.pt'))
    print('Model loaded.')


##################################################################################
# 推理
def predict():
    fp = open('test_prediction.txt', 'w')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader)):
            ids = batch['input_ids'].to(config['device'], dtype=torch.long)
            mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
            tr_logits = model(input_ids=ids, attn_masks=mask)
            preds = tr_logits.detach().cpu().numpy()
            pred_class = np.argmax(preds, axis=2)
            for i in range(len(pred_class)):
                for j in range(len(batch['labels'][i])):
                    if batch['labels'][i][j] != -100:
                        fp.write(batch['text'][i][j] + ' ' + ids_to_labels[pred_class[i][j]] + '\n')
                fp.write('\n')
    fp.close()


predict()
