# 导入配置
from config import CFG
from datapreprocess import load_glove
from weight import cal_weight

import os
import numpy as np
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification,AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# 从hugging face上加载预训练模型
class MentalRoberta(nn.Module):
    def __init__(self,model_path,num_classes,use_auth_token):
        super(MentalRoberta,self).__init__()
        if (isinstance(model_path,str) is False) or (isinstance(num_classes,int) is False) or (isinstance(use_auth_token,str) is False):
            raise TypeError("Type Error!")
        # 设置参数
        self.model_path = model_path
        self.num_classes = num_classes  
        self.use_auth_token = use_auth_token

        # 加载预训练模型分类器
        print(f"Load {self.model_path} pretrained model classifier!")
        self.roberta = BertForSequenceClassification.from_pretrained(self.model_path,
                                                                          use_auth_token  = self.use_auth_token,
                                                                          num_labels      = self.num_classes)
        
        for param in self.roberta.parameters():
            param.requires_grad = False
        # 增加线性分类层,in_features
        in_features = self.roberta.classifier.in_features
        self.roberta.classifier = nn.Sequential(
            nn.Linear(in_features,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048,self.num_classes)
        )
        self.roberta.classifier.requires_grad = True

    def forward(self,x,attention_mask=None,labels=None,mode=None):
        """
        前向传播

        Parameters:
        -----------
        x: 输入的数据, shape-> [batch,maxlen]
        attention_mask: padding的位置记录，让模型避免在padding上增加注意力
        labels: 真实标签
        """
        if mode in ["train","val","test"]:
            y_ = self.roberta(x, token_type_ids=None, attention_mask = attention_mask, labels = labels)
        else:
            y_ = self.roberta(x, token_type_ids=None)
        return y_

# BiLSTM用于精神疾病分类
class BiLSTM(nn.Module):
    def __init__(self,n_classes,embed_size,embedding_matrix):
        super(BiLSTM, self).__init__()
        self.hidden_size = CFG.hidden_layer
        self.n_classes = n_classes
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size,num_layers = 2, bidirectional=True, batch_first=True, dropout=0.2)

        ## 分类层 ##
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*4,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048,self.n_classes)
        )


    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        out = self.classifier(conc)
        return out


class TextCNN(nn.Module):
    def __init__(self,embedding_matrix,embed_size,number_classes):
        super(TextCNN, self).__init__()
        #kernel size will be filter _size * embedding size
        # we will have 5 filter covering these many words at a time
        self.filter_1 = 1 
        self.filter_2 = 2
        self.filter_3 = 3
        self.filter_4 = 4
        self.filter_5 = 5
        num_filters = 15 # no of output channels
        self.number_classes = number_classes
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix,freeze=True)
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.conv_1 = nn.Conv2d(1,num_filters,(self.filter_1, embed_size))
        self.conv_2 = nn.Conv2d(1,num_filters,(self.filter_2, embed_size))
        self.conv_3 = nn.Conv2d(1,num_filters,(self.filter_3, embed_size))
        self.conv_4 = nn.Conv2d(1,num_filters,(self.filter_4, embed_size))
        self.conv_5 = nn.Conv2d(1,num_filters,(self.filter_5, embed_size))
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(5*num_filters, self.number_classes)

    def forward(self, x):
        x = self.embedding(x)  
        x =  x.unsqueeze(1)
        x1 =  F.max_pool1d( F.relu(self.conv_1(x)).squeeze(3) , F.relu(self.conv_1(x)).squeeze(3).size(2)).squeeze(2)
        x2 =  F.max_pool1d( F.relu(self.conv_2(x)).squeeze(3) , F.relu(self.conv_2(x)).squeeze(3).size(2)).squeeze(2)
        x3 =  F.max_pool1d( F.relu(self.conv_3(x)).squeeze(3) , F.relu(self.conv_3(x)).squeeze(3).size(2)).squeeze(2)
        x4 =  F.max_pool1d( F.relu(self.conv_4(x)).squeeze(3) , F.relu(self.conv_4(x)).squeeze(3).size(2)).squeeze(2)
        x5 =  F.max_pool1d( F.relu(self.conv_5(x)).squeeze(3) , F.relu(self.conv_5(x)).squeeze(3).size(2)).squeeze(2)
        x = torch.cat((x1,x2,x3,x4,x5),1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x
    

# Focal Loss
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[1/6 for i in range(CFG.num_classes)], gamma=2, reduction='mean'):
        """
        function:
        ---------
            focal loss实现
        
        args:
        ---------
            alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
            gamma: 困难样本挖掘的gamma
            reduction: optional['mean','sum']
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(CFG.device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        function:
        ---------
            计算focal loss损失.
        
        args:
        ---------
            pred: 模型预测值
            target: 真实标签
        
        returns:
        ---------
            focal_loss: 返回计算好的focal loss张量

        """
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss    

def Load_System(args,train_csv=None,test_csv=None):
    """
    function:
    ---------
        加载模型结构，损失函数以及分词工具.
    
    args:
    ---------
        args: 系统参数
        train_csv: 训练数据.
        test_csv[]: 测试数据.
    
    returns:
    ---------
        System: 返回模型、损失函数以及分词工具.
    """

    assert args.loss.lower() in ["bce","focal"], "Unexpected Loss"
    if args.m == "roberta":
        if args.inplace == "huggingface":
            model = MentalRoberta(model_path     = CFG.bert_base_uncased,
                                  num_classes    = CFG.num_classes,
                                  use_auth_token = CFG.use_auth_token)
            

            # 加载Roberta对应的tokenzier
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = CFG.bert_base_uncased,
                                                        use_auth_token                = CFG.use_auth_token)
            # 保存加载好的模型
            torch.save(model,"mental-bert-base-uncased.pth")
            torch.save(tokenizer,"tokenizer-mental-bert-base-uncased.pth")
        
        elif args.inplace == "local":
            model , tokenizer = torch.load(args.mfile,map_location=CFG.device) , torch.load(args.tfile,map_location=CFG.device)

        model = model.to(CFG.device)
        criterion = None
    
    elif args.m == "bilstm" or args.m == "textcnn":
        # 生成词表
        print("Create Vocab...")
        tokenizer = Tokenizer(num_words=CFG.max_features)
        tokenizer.fit_on_texts(list(train_csv.df["corpus"]) + list(test_csv.df["corpus"]))
        file = open(f"tokenizer-{args.m}.pickle","wb")
        pickle.dump(tokenizer,file)
        file.close()

        print("Load glove pretrained vector...")
        # 加载预训练的glove词向量模型
        if os.path.exists(args.glove):
            embedding_matrix = torch.tensor(load_glove(word_index = tokenizer.word_docs,
                                                        max_features = CFG.max_features, 
                                                        EMBEDDING_FILE = args.glove),dtype=torch.float)
        else:
            embedding_matrix = torch.tensor(np.random.randn(120000,args.embed))
        
        print("embeding matrix shape: ",embedding_matrix.shape," dtype: ",embedding_matrix.dtype)
        # 定义bilstm模型
        if args.m == "bilstm":
            model = BiLSTM(n_classes = CFG.num_classes,embed_size = CFG.embed_size, embedding_matrix = embedding_matrix)
        elif args.m == "textcnn":
            model = TextCNN(number_classes = CFG.num_classes, embed_size = CFG.embed_size, embedding_matrix = embedding_matrix)
        
        model = model.to(CFG.device)

        if args.loss == "bce":
        # 定义损失函数 [2.6989459 ,25.47729149,25.63606149,3.77776417,3.93081921,31.13319879]
            criterion = nn.CrossEntropyLoss(weight=cal_weight(list(train_csv.df["class_id"].value_counts()),method=args.weight),reduction='mean').to(CFG.device)
        elif args.loss == "focal":
            criterion = MultiClassFocalLossWithAlpha().to(CFG.device)

    System = {
        "model": model,
        "name": args.m,
        "cal_loss":criterion,
        "tokenizer": tokenizer
    }

    return System