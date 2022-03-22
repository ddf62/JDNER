import json

from processors.ner_seq import ner_processors as processors

lines = []
processor = processors["jdner"]()
examples = processor.get_test_examples('./datasets/JDNER')
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
