from Model import Load_System
from config import CFG
from datapreprocess import DataPipline
from error import NumberOfDeviceError
from dataset import MentalHealthDataset
from train import Training
from test import testing

import torch.optim as optim
import torch
import warnings
import os
import argparse
import pickle
warnings.filterwarnings("ignore")
 
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer



if __name__ == "__main__":
    num_gpu = torch.cuda.device_count()
    parse = argparse.ArgumentParser(description="Demo.py Parameters")

    # 必须参数,train代表训练模式, eval代表测试/查询模式
    parse.add_argument("--mode",help="The mode in which the module runs, optional ['train','eval']",type=str,default="eval")
    parse.add_argument("--m",help="The model you want to train, optional[roberta,bilstm,textcnn]",type=str,default="roberta")

    # 通用训练参数
    parse.add_argument("--lr",help="learning rate",type=float,default=1e-3)
    parse.add_argument("--batch",help="batch size",type=int,default=1)
    parse.add_argument("--epoch",help="The number of iterations of training",type=int,default=5)
    parse.add_argument("--g",help="Which GPU do you want to train the model on.",type=int,default=0)
    parse.add_argument("--maxlen",help="The maximum length of each piece of data,default=512",type=int,default=512)
    # parse.add_argument("--vs",help="split train set and valid set",type=int,default=0.15)
    parse.add_argument("--weight",help="set class weight using different method",type=int,default=0)
    parse.add_argument("--loss",help="Which loss do you want to use",type=str,default="bce")
    parse.add_argument("--K",help="K-fold Cross Validation",type=int,default=5)

    # BiLSTM特有参数
    parse.add_argument("--hidden",help="numbers of hiddern layers in BiLSTM",type=int,default=64)
    parse.add_argument("--mv",help="Maximum allowed length of vocabulary",type=int,default=200000)
    parse.add_argument("--embed",help="how big is each word vector, optional: [50,100,200,300]",type=int,default=100)
    parse.add_argument("--glove",help="Glove Vector",type=str,default="./glove.6B.100d.txt") 
    # parse.add_argument("--d",help="the file which store dict",type=str,default="")

    #Roberta特有参数
    parse.add_argument("--inplace",help="Load pre trained models from local or hugging faces",type=str,default="local")
    parse.add_argument("--mfile",help="The file path of model.pth",type=str,default="mental-bert-base-uncased.pth")
    parse.add_argument("--tfile",help="The file path of tokenizer.pth",type=str,default="tokenizer-mental-bert-base-uncased.pth")

    # 文件路径参数
    # parse.add_argument("--train",help="The file path of train csv",type=str,default="./Reddit Dataset/both_train.csv")
    # parse.add_argument("--val",help="The file path of valid csv",type=str,default="./Reddit Dataset/both_val.csv")
    # parse.add_argument("--test",help="The file path of test csv",type=str,default="./Reddit Dataset/both_test.csv")
    parse.add_argument("--outputdir",help="save output for training",type=str,default="./outputs/")



    args = parse.parse_args()
    if (num_gpu == 0 and args.g != -1) or (num_gpu > 0 and args.g > num_gpu) or (num_gpu < -1):
        raise NumberOfDeviceError(num_gpu,args.g)
    
    # 修改config模块内的参数
    CFG.batch_size = args.batch
    CFG.lr = args.lr
    CFG.EPOCH = args.epoch
    CFG.max_len = args.maxlen
    CFG.model_name = args.m
    CFG.embed_size = args.embed
    CFG.hidden_layer = args.hidden
    CFG.max_features = args.mv
    if os.path.exists(args.outputdir) is False:
        os.mkdir(args.outputdir)
    CFG.outputdir = args.outputdir

    # 判断当前的gpu是否合理
    if args.g == -1:
        CFG.device = torch.device("cpu")
    elif num_gpu > 0 and args.g != -1:
        CFG.device = torch.device(f"cuda:{args.g}" if torch.cuda.is_available() else "cpu")


    print(f"Device:{torch.cuda.get_device_name(torch.cuda.current_device)} will be used!")
    # 训练模式
    if args.mode == "train":
        print("Model will be trained using these hyper-parameters [lr: {:3f}] [batch-size:{:3d}] [EPOCH: {:2d}] [maxlen: {:4d}]".format(args.lr,
                                                                                                                                        args.batch,
                                                                                                                                        args.epoch,
                                                                                                                                        args.maxlen))
        # 加载训练数据集和验证数据集
        train_csv = DataPipline(mode="train")
        test_csv  = DataPipline(mode="test")
        # 数据预处理
        train_csv.preprocess()
        test_csv.preprocess()

        # K折交叉验证
        kf = KFold(n_splits=args.K,shuffle=True,random_state=CFG.seed_val)
        KFold_Datas = kf.split(train_csv.df)
        
        for k_index,(train_indexs,valid_indexs) in enumerate(KFold_Datas):
            print(f"**********************{k_index}Fold:")
            # 定义模型、损失函数和分词
            system = Load_System(args      = args, 
                                train_csv = train_csv,
                                test_csv  = test_csv)
            
            
            # 定义优化器和训练盒子
            optimizer = optim.AdamW(system["model"].parameters(),lr=CFG.lr,eps = 1e-8,weight_decay=2e-4)
            train_box = Training(model = system["model"] , modelname = args.m)

            # 加载数据集(包括训练集、验证集和测试集)
            train_df = train_csv.df.iloc[train_indexs].reset_index()
            valid_df = train_csv.df.iloc[valid_indexs].reset_index()

            Training_Dataset = MentalHealthDataset(df = train_df, tokenizer = system["tokenizer"],mode = args.m)
            Validation_Dataset = MentalHealthDataset(df = valid_df, tokenizer = system["tokenizer"],mode = args.m)
            Test_Dataset = MentalHealthDataset(df = test_csv.df,tokenizer = system["tokenizer"], mode = args.m)

            train_loader = DataLoader(dataset = Training_Dataset, batch_size = CFG.batch_size, shuffle = True)
            valid_loader = DataLoader(dataset = Validation_Dataset,batch_size = CFG.batch_size, shuffle = True)
            test_loader = DataLoader(dataset = Test_Dataset, batch_size = CFG.batch_size, shuffle = True)


            # 定义学习率递减策略，roberta默认使用warmup, bilstm/textcnn默认使用余弦退火算法
            if(args.m == "roberta"):
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 100, num_training_steps = len(train_loader) * CFG.EPOCH)
            elif(args.m == "bilstm" or args.m == "textcnn"):
                scheduler = CosineAnnealingLR(optimizer,T_max = CFG.EPOCH)

            # 训练以及验证
            training_metrics, valid_metrics = train_box.training(train_loader = train_loader,
                                                                 valid_loader = valid_loader,
                                                                 optimizer    = optimizer, 
                                                                 scheduler    = scheduler,
                                                                 criterion    = system["cal_loss"])
            
            # 测试集测试
            accuracy , f1 , loss = testing(test_loader = test_loader,
                                           model       = system["model"],
                                           criterion   = system["cal_loss"],
                                           args        = args,
                                           kfold       = k_index)

            # 每次循环保存一次指标
            metrics = {
                "train":training_metrics,
                "valid":valid_metrics,
                "test_acc":accuracy,
                "test_f1": f1,
                "test_loss": loss
            }   
            filename = args.outputdir + f"{args.m}-fold{k_index}-metrics.pickle"
            file = open(filename,"wb")
            pickle.dump(metrics,file)
            file.close()

            # 保存模型
            checkpoints = {
                "model": system["model"].state_dict(),
                "tokenizer": system["tokenizer"],
                "name":args.m,
            }
            model_path = args.outputdir + f"{args.m}-fold{k_index}-model.pth"
            torch.save(checkpoints,model_path)
            print(f"Model have been established! [FILE: {model_path}]")
        
    elif args.mode == "test":
        pass    
    elif args.mode == "eval":
        print("Start Evaling!") 
    
