# baseline说明

本baseline为GAIIC 2022 赛题2-商品标题实体识别 参赛选手提供参考，基于开源代码修改（https://github.com/lonePatient/BERT-NER-Pytorch）

## 模型

1. bert+crf

## 配置

1. 预训练模型以及词表位置：启动脚本中默认配置为项目目录下的prev_trained_model下，可根据需要自行修改。
2. 数据：在processors/new_seq.py中默认数据集文件名为train.txt、dev.txt、test.txt，可根据需要自行修改。
3. 输出路径，启动脚本默认配置为项目目录下的outputs。

---

## 训练及验证

1. 启动脚本位于scripts下，run_ner_crf.sh为边训边验证，`MODEL_NAME_OR_PATH`指向原始bert，eval_ber_crf.sh为仅仅验证，`MODEL_NAME_OR_PATH`指向训练后的模型，参赛者可以根据需要自行调节`--do_train`,`--do_eval`,`--do_predict`参数


---
