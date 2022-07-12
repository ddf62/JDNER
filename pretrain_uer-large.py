import os
import csv
from tqdm import tqdm
from transformers import  BertTokenizer, WEIGHTS_NAME, BertTokenizerFast,TrainingArguments
import tokenizers
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)
from model.modeling_nezha import NeZhaForMaskedLM
from model.configuration_nezha import NeZhaConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
MODEL_CLASSES = {
    'nezha': (NeZhaConfig, NeZhaForMaskedLM, BertTokenizer)
}
## 加载tokenizer和模型
config_class, model_class, tokenizer_class = MODEL_CLASSES['nezha']

tokenizer =  BertTokenizerFast.from_pretrained('./prev_trained_model/nezha-cn-base', do_lower_case=True)
config = config_class.from_pretrained('./prev_trained_model/nezha-cn-base')
model = model_class.from_pretrained('./prev_trained_model/nezha-cn-base', config=config)
tokenizer.add_tokens(' ')
model.resize_token_embeddings(len(tokenizer))
# 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
def read_test_text(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f.read().split('\n'):
            lines.append({"words": list(line)})
    return lines[:-1]


def preprocessor(df, tokenizer):
    pad = tokenizer.pad_token
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    # GET TEXT AND WORD LABELS
    for index in tqdm(range(len(df))):
        data = df[index]
        text = [i.lower() for i in data['words']]
        if len(text) > 126:
            text = text[:126]
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
        df[index] = {'input_ids': input_ids}
    return df


class dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        # GET TEXT AND WORD LABEL
        return self.data[index]

    def __len__(self):
        return self.len

    
test_text = read_test_text('datasets/JDNER/unlabeled_train_data.txt')
print(len(test_text))
test_df = preprocessor(test_text, tokenizer)
train_dataset = dataset(test_df)
# train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='datasets/JDNER/unlabeled_train_data.txt',block_size=128) 
# MLM模型的数据DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.4)

# 训练参数
pretrain_batch_size=128
num_train_epochs=40
training_args = TrainingArguments(
    output_dir='./prev_trained_model/nezha-mlm', overwrite_output_dir=True, num_train_epochs=num_train_epochs, learning_rate=2e-5,
    per_device_train_batch_size=pretrain_batch_size, save_total_limit=5)# save_steps=10000
# 通过Trainer接口训练模型
trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

# 开始训练
trainer.train(False)
trainer.save_model('./prev_trained_model/nezha-mlm-0.4')

