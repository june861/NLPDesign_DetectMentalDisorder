from config import CFG

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import random

from nltk import word_tokenize
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor



class DataPipline(object):
    def __init__(self,mode,n=None) -> None:
        assert mode.lower() in ["train","val","test"], "Error Parameter!"

        self.mode = mode
        self.n = n
        if mode == "train":
            self.df1 = pd.read_csv("./Reddit Dataset/ptsd.csv")
            self.df2 = pd.read_csv("./Reddit Dataset/adhd.csv")
            self.df3 = pd.read_csv("./Reddit Dataset/depression.csv")
            self.df4 = pd.read_csv("./Reddit Dataset/mental_disorders_reddit.csv")

            self.df5 = pd.read_csv("./Reddit Dataset/both_train.csv")
            self.df6 = pd.read_csv("./Reddit Dataset/both_val.csv")
            self.df = None
        else:
            self.df = pd.read_csv("./Reddit Dataset/both_test.csv")
    
    def Train_Data_Preprocess(self):
        """
        预处理训练数据集
        """
        frames = []

        self.df1 = self.df1[self.df1["body"] != "[deleted]"][self.df1["body"] != "[removed]"]
        self.df2 = self.df2[self.df2["body"] != "[deleted]"][self.df2["body"] != "[removed]"]
        self.df3 = self.df3[self.df3["body"] != "[deleted]"][self.df3["body"] != "[removed]"]
        self.df4 = self.df4[self.df4["selftext"] != "[deleted]"][self.df4["selftext"] != "[removed]"]

        self.df1 = self.df1[self.df1["title"] != "[deleted]"][self.df1["title"] != "[removed]"]
        self.df2 = self.df2[self.df2["title"] != "[deleted]"][self.df2["title"] != "[removed]"]
        self.df3 = self.df3[self.df3["title"] != "[deleted]"][self.df3["title"] != "[removed]"]
        self.df4 = self.df4[self.df4["title"] != "[deleted]"][self.df4["title"] != "[removed]"]

        # 去除NAN值
        self.df1.dropna(inplace=True)
        self.df2.dropna(inplace=True)
        self.df3.dropna(inplace=True)
        self.df4.dropna(inplace=True)

        self.df1.drop(columns=["created_utc","url","author","id","num_comments","score","upvote_ratio"],inplace=True)
        self.df2.drop(columns=["created_utc","url","author","id","num_comments","score","upvote_ratio"],inplace=True)
        self.df3.drop(columns=["created_utc","url","author","id","num_comments","score","upvote_ratio"],inplace=True)
        self.df4.drop(columns=["created_utc","over_18"],inplace=True)

        self.df1.insert(loc=2,column="class_id",value=[CFG.classes.index(self.df1["subreddit"][0].lower()) for i in range(len(self.df1))])
        self.df2.insert(loc=2,column="class_id",value=[CFG.classes.index(self.df2["subreddit"][0].lower()) for i in range(len(self.df2))])
        self.df3.insert(loc=2,column="class_id",value=[CFG.classes.index(self.df3["subreddit"][0].lower()) for i in range(len(self.df3))])

        self.df1.rename(columns={"body":"post","subreddit":"class_name"},inplace=True)
        self.df2.rename(columns={"body":"post","subreddit":"class_name"},inplace=True)
        self.df3.rename(columns={"body":"post","subreddit":"class_name"},inplace=True)
        self.df4.rename(columns={"subreddit":"class_name","selftext":"post"},inplace=True)

        if isinstance(self.n,int) is False:
            self.n = int((len(self.df1) + len(self.df2) + len(self.df3)) / 3)
        
        random_state = random.randint(0,10086)
        # 获取一部分的bipoar和anxiety的数据 
        anxiety = self.df4[self.df4["class_name"] == "Anxiety"].sample(n=self.n,axis=0,random_state=random_state).reset_index()
        bipolar = self.df4[self.df4["class_name"] == "bipolar"].sample(n=self.n,axis=0,random_state=random_state).reset_index()
        

        anxiety.insert(loc=2,column="class_id",value=[CFG.classes.index("anxiety") for i in range(len(anxiety))])
        bipolar.insert(loc=2,column="class_id",value=[CFG.classes.index("bipolar") for i in range(len(bipolar))])

        frames.append(self.df1[["post","class_id","class_name","title"]])
        frames.append(self.df2[["post","class_id","class_name","title"]])
        frames.append(self.df3[["post","class_id","class_name","title"]])
        frames.append(anxiety[["post","class_id","class_name","title"]])
        frames.append(bipolar[["post","class_id","class_name","title"]])
        frames.append(self.df5[["post","class_id","class_name","title"]])
        frames.append(self.df6[["post","class_id","class_name","title"]])

        self.df = pd.concat(frames,ignore_index=True)
        self.df["corpus"] = self.df["title"] + self.df["post"]
        
    def Test_Data_Preprocess(self):
        self.df["corpus"] = self.df["title"] + self.df["post"]

    def preprocess(self):
        if self.mode == "train":
            self.Train_Data_Preprocess()
            print("**************Training Datas:")
        else:
            self.Test_Data_Preprocess()
            print("**************Testing Datas:")
        print(self.df.head,end="\n")



## Load Glove Vector
def load_glove(word_index,max_features,EMBEDDING_FILE = './glove/glove.6B.100d.txt'):
    """
    functions:
    ----------
        依据输入的词典加载预训练好的Glove词向量

    args:
    ----------
        word_index: 输入的词典
        max_features: 词典允许的最大尺寸
        EMBEDDING_FILE: Glove词向量加载文件(.txt)，默认为./glove/glove.6B.100d.txt
    
    returns:
    ----------
        embedding_matrix: 词嵌入矩阵
    """
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf-8"))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix
