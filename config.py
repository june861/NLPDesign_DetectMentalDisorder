import torch

class CFG:
    #------------------------------Public Parameters-------------------------------#
    # 01. EPOCH: 迭代次数，默认为5                                                  
    # 02. lr: 学习率，默认为2e-5                       
    # 03. batch_size : 小批量大小，默认为1
    # 04. max_len : 在训练过程中每条数据允许的最大的长度
    # 05. padding_value: 填充的数值，默认为0
    # 06. use_auth_token: hugging face的密钥
    # 07. classes: 标签类别，在本次任务中一共有6个类别
    # 08. num_classes: 标签数
    # 09. device: 设备名，通常指cuda或者cpu
    # 10. seed_val: 随机种子，固定随机种子，消除随机性带来的误差
    # 11. model_name: 模型名称，本次实验中只有以下选项["bilstm","roberta","textcnn"]
    # 12. outputdir: 输出地址
    #------------------------------------------------------------------------------#
    EPOCH = 5
    lr = 2e-5 
    batch_size = 1
    max_len = 512
    padding_value = 0
    use_auth_token = "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"
    classes = ["adhd","anxiety","bipolar","depression","ptsd","none"]
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_val = 42
    model_name = ""
    outputdir = "./outputs/"

    #-------------------------------Only BiLSTM Used-------------------------------#
    # 01. hidden_layer: BiLSTM隐藏层层数，默认为64
    # 02. embed_size: 词嵌入维度,[50,100,200,300]，默认为100
    # 03. max_features: 词典最大词数，默认为120000
    #------------------------------------------------------------------------------#
    hidden_layer = 64
    embed_size = 100
    max_features = 120000

    #---------------------Pretrained Model in HuggingFace--------------------------#
    # 01. mental_bert_uncased: 经过mental-health预训练好的roberta模型
    #------------------------------------------------------------------------------#
    bert_base_uncased = "bert-base-uncased"
    mental_bert_uncased = "mental/mental-bert-base-uncased"
    mental_bert = "mental/mental-roberta-base"
    
