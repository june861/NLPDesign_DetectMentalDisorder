# Readme

------

## Background

本次项目是用于2023北京邮电大学的自然语言处理课设，课程设计的主要内容是基于国外社交平台(reddit、twitter)的精神疾病分类。

## Dependency

>Linux n1 4.18.0-80.7.1.el8_0.x86_64
>
>pytorch 2.0.1+cu117
>
>tensorflow 2.12.0
>
>scikit-learn 1.2.2
>
>gradio 3.34.0
>
>numpy 2.8.4
>
>pandas 2.0.2

## Author

2020212236	罗玮俊

2020212234	李云川

2020212248	周文卿

## Project

### 模型训练

本次实验在单机单卡的环境下进行训练，项目文件结构如下所示(代码文件全部在根目录下)。其中`demo.py`文件内置了多个可选择参数，参数的具体作用如下所示：

| 参数名称    | 参数作用                                                     | 默认值     |
| ----------- | ------------------------------------------------------------ | ---------- |
| `mode`      | 模式选择，通常有`train`,`test`以及`eval`                     | 'eval'     |
| `m`         | 模型名称，参数可选项为`roberta`,`bilstm`以及`textcnn`        | 'roberta'  |
| `lr`        | 模型训练时的学习率                                           | 0.001      |
| `batch`     | 小批量大小                                                   | 1          |
| `epoch`     | 训练的迭代次数                                               | 5          |
| `g`         | 使用哪块gpu进行训练                                          | -1         |
| `maxlen`    | 每条数据允许的最大长度                                       | 512        |
| `weight`    | 类别权重，共有4中方法，可选项为[0,1,2,3]                     | 0          |
| `loss`      | 损失函数选择，可选参数['bce','focal']                        | 'bce'      |
| `K`         | k折交叉验证                                                  | 5          |
| `hidden`    | BiLSTM的隐藏层结点数                                         | 64         |
| `mv`        | 词表允许的最大长度(本参数只在训练bilstm和textcnn时生效)      | 200000     |
| `embed`     | `Glove`词向量的特征维度，可选项为[50,100,200,300]            | 100        |
| `glove`     | `Glove`词向量文件的路径，当`embed`不为100时必须给出          | --         |
| `inplace`   | 加载`Roberta`模型，从本地或者huggingface,可选项['local','huggingface'],第一次运行项目时必须使用`huggingface`参数 | local      |
| `mfile`     | 模型文件所在路径，当`inplcae=local`时才生效                  | --         |
| `tfile`     | tokenizer所在的路径，当`inplcae=local`时才生效               | --         |
| `outputdir` | 输出路径                                                     | ./outputs/ |

如果你想训练一个`bilstm`模型，可以使用如下命令进行训练:

```bash
python demo.py --m bilstm --mode train --batch 128 --epoch 10 --hidden 32 --g 0 | tee bilstm.log
```

如果你想训练一个[roberta](https://huggingface.co/mental/mental-bert-base-uncased)预训练模型(预训练模型来自huggingface的预训练模型)可以使用如下的命令：

```bash
python demo.py --m roberta --mode train --batch 128 --epoch 10 --g 0 | tee roberta.log
```

如果你想训练一个`textcnn`模型，可以使用如下参数进行训练：

```bash
 python demo.py --m textcnn --mode train --batch 128 --epoch 20 --g 0 | tee textcnn.log
```

 

### UI界面运行

在根目录下直接使用下面命令:

```bash
python UI.py
```

在运行是必须确保根目录下存在已经训练好的模型文件(.pth)



### 数据资源下载

数据集已经上传至
