# JDNER, 第一，第一，还是他妈的第一
### 数据分析：https://docs.qq.com/sheet/DQ2J0cWlKVVNsS2pt

###问题
1. 组成句子再分词，会存在实体丢失的问题。
2. 
|       | 模型                                                    | dev                           | test     | 干啥了                                                                                             |
|-------|-------------------------------------------------------|-------------------------------|----------|-------------------------------------------------------------------------------------------------|
| 03-22 | baseline                                              | 0.7885                        | 0.785504 | baseline                                                                                        |
| 03-22 | baseline-1                                            | 0.7823                        | 0.78604  | 调低了学习率，增大了epoch                                                                                 |
| 03-23 | baseline-roberta                                      | 0.7890(3150)                  | 0.7854   | bert to roberta                                                                                 |
| 03-23 | roberta-v1                                            | 0.7884（4275）                  | 0.7843   | 连接后四层                                                                                           |
| 03-28 | bert-base-globalpointer(zhr-李老师服务器)                   | 0.7911                        | 0.75982  | 使用了 [源码](https://github.com/gaohongkui/GlobalPointer_pytorch) ,在处理数据时，删去了空格,且为整句tokenize        |
| 03-28 | bert-base-globalpointer（zhr-nlp服务器）                   | 0.7487                        |          | 使用了重写版本，将大写英文全部改为小写，保留空格，并在bert字典里添加了空格                                                         |
| 03-28 | bert-base-globalpointer（广哥服务器）                        | 0.682338                      |          | 较上一个版本添加了dropout，修改warmup为get_polynomial_decay_schedule_with_warmup(30个epoch，但是过拟合还不严重）         |
| 03-28 | ernie-2.0-globalpointer(zhr-nlp服务器)                   |                               |          | 依然使用cos_warmup,添加dropout，rate为0.22                                                              |
| 03-28 | globalpointer-baseline(广哥服务器）                         | 0.8452097116704104(在后4000上验证） | 0.80266  | 用了讨论区[baseline](https://github.com/DataArk/GAIIC2022-Product-Title-Entity-Recognition-Baseline) |
| 03-28 | chinese-bert-wwm-globalpointer(zhr-nlp服务器)            | 0.75619                       |          | 使用重写版本cos warmup，dropout率为0.22，但30个epoch还未收敛                                                    |
| 03-29 | ernie1.0-globalpointer(zhr-nlp服务器)                    | 0.7589031699347775            |          | 设置同上                                                                                            |
| 03-29 | ernie1.0-globalpointer-baseline(广哥服务器)                | 0.8241530740276035            |          | 把讨论区baseline模型换成ernie1.0(5个epoch)                                                               |
| 03-29 | chinese-roberta-wwm-ext-globalpointer-baseline(广哥服务器) | 0.841652                      | 0.801785 | 同上                                                                                              |
| 03-29 | chinese-bert-wwm-globalpointer-baseline(广哥服务器)        | 0.8036865                     | 0.79859  | 讨论区版，train和dev集换成我们划的                                                                           |
| 03-29 | bert-base-chinese-globalpointer-seq                   | 0.70504                       |          | 将单个字合并成一个序列进行tokenize,10个epoch，调整实体映射方法                                                         |
| 03-30 | bert-base-chinese-globalpointer                       |                               |          | 同03-28-2，去掉warmup                                                                               |
| 03-30 | bert-base-globalpointer-baseline                      | 0.86766                       | 0.8012   | bert-wwm换成bert-base                                                                             |


##03-31 重大突破 GlobalPointer改对了
|       | 模型                          | dev      | test      | 做啥了                                       |
|-------|-----------------------------|----------|-----------|-------------------------------------------|
| 03-31 | uer-large-globalponter-seq  | 0.805381 | 0.80187   | 新版seq globalpointer，没有warmup，用AutoModel加载 |
| 04-01 | uer-large-globalpointer-seq | 0.81407 | 0.807206  | 用了cos的warm up |
 | 04-01 | uer-large-globalpointer-seq | 0.810139 |           | 将后四层连接，修改warmup为get_polynomial_decay_schedule_with_warmup|
 | 04-01 | uer-large-globalpointer-seq | 0.85553 |  0.80616  | 用了所有的训练集。warmup同上|                  