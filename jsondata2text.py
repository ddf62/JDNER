import os
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

import json


lines = []
examples = get_test_examples('./datasets/JDNER')
fp = open('./outputs/JDNER_output_eval/bert/test_prediction.json', 'r', encoding='utf-8')
dict = fp.read().split('\n')
print(len(examples))
with open('test_predict.txt', 'w') as ff:
    for i in range(len(examples)):
        json_data = json.loads(dict[i])
        json_data['tag_seq'] = json_data['tag_seq'].split(' ')
        for j in range(len(examples[i].text_a)):
            ff.write(examples[i].text_a[j] + ' ' + json_data['tag_seq'][j] + '\n')
        ff.write('\n')

