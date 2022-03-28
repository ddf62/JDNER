# JDNER, 第一，第一，还是他妈的第一
### 数据分析：https://docs.qq.com/sheet/DQ2J0cWlKVVNsS2pt
|       | 模型                                  | dev        | test    | 干啥了                                                                                      |
|-------|-------------------------------------|------------|---------|------------------------------------------------------------------------------------------|
| 03-22 | baseline                            | 0.7885     | 0.785504 | baseline                                                                                 |
| 03-22 | baseline-1                          | 0.7823     | 0.78604 | 调低了学习率，增大了epoch                                                                          |
| 03-23 | baseline-roberta                    | 0.7890(3150) | 0.7854  | bert to roberta                                                                          |
| 03-23 | roberta-v1                          | 0.7884（4275） | 0.7843  | 连接后四层                                                                                    |
| 03-28 | bert-base-globalpointer(zhr-李老师服务器) | 0.7911     | 0.75982 | 使用了 [源码](https://github.com/gaohongkui/GlobalPointer_pytorch) ,在处理数据时，删去了空格,且为整句tokenize |
| 03-28 | bert-base-globalpointer（zhr-nlp服务器） | 0.7487     |         | 使用了重写版本，将大写英文全部改为小写，保留空格，并在bert字典里添加了空格                                                  |
| 03-28 | bert-base-globalpointer（广哥服务器）|            |         | 较上一个版本添加了dropout，修改warmup为get_polynomial_decay_schedule_with_warmup                                                               |
| 03-28 | ernie-2.0-globalpointer(zhr-nlp服务器)| | | 依然使用cos_warmup,添加dropout，rate为0.22 |
| 03-28 | globalpointer-baseline(广哥服务器）| | | 用了讨论区[baseline](https://github.com/DataArk/GAIIC2022-Product-Title-Entity-Recognition-Baseline)|