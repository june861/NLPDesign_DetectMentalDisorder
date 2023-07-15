import pandas as pd
import torch
import re
from config import CFG
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

special_symbols = [',', '.', '"', ':', ')', '(', '-', '!', '?', '\n','|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£','·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
pattern = "[" + re.escape("".join(special_symbols)) + "]"

class MentalHealthDataset(Dataset):
    def __init__(self,df,tokenizer=None,mode=None) -> None:
        if isinstance(mode,str) is False:
            raise TypeError("parameters: --mode must be a string type!")
        if mode.lower() not in ["roberta","bilstm","textcnn"]:
            raise ValueError("Parameter: --mode must be in [robert, bilstm]")
        self.mode = mode.lower()
        self.df = df
        self.tokenizer = tokenizer


    def __getitem__(self, item):
        """
        返回指定数据
        """
        # 处理字符串，去除特殊字符
        text = re.sub(pattern, '',self.df["corpus"][item])
        if self.mode == "roberta":
            if self.tokenizer == None:
                raise TypeError("NoneType can't not be used!")
            # 计算ID, List --> [[id1,id2,....,idn]]
            inputIds = [self.tokenizer.encode(text, add_special_tokens= True)]
            # 填充序列
            inputIdsTrunc = pad_sequences(inputIds, maxlen = CFG.max_len, 
                                          dtype = "long", value = CFG.padding_value, 
                                          truncating = "post", padding = "post")
            # 计算mask，模型不应该关注填充部分的内容
            attentionMasks = [int(tokenId>0) for tokenId in inputIdsTrunc[0]]
            # 获取label
            labels = self.df["class_id"][item]
            # 数据类型转化
            X , Y , mask = torch.tensor(inputIdsTrunc[0]) , torch.tensor(labels) , torch.tensor(attentionMasks)
            return X,mask,Y
        
        elif self.mode == "bilstm" or self.mode == "textcnn":
            if self.tokenizer == None:
                raise TypeError("NoneType can't not be used!")
            # 计算ID
            # 字符串预处理，去除空行
            
            inputIds = self.tokenizer.texts_to_sequences([text])
            # 填充序列
            inputIdsTrunc = pad_sequences(inputIds, maxlen = CFG.max_len, 
                                dtype = "long", value = CFG.padding_value, 
                                truncating = "post", padding = "post")
            # 获取labels
            labels = self.df["class_id"][item]

            X , Y = torch.tensor(inputIdsTrunc,dtype=torch.int) , torch.tensor(labels,dtype=torch.int)
            return X,Y

    def __len__(self):
        return len(self.df)