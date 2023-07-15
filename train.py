from Model import MentalRoberta,BiLSTM,TextCNN
from config import CFG

import numpy as np
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score , classification_report

np.random.seed(CFG.seed_val)
torch.manual_seed(CFG.seed_val)
torch.cuda.manual_seed_all(CFG.seed_val)



class Validation(object):
    def __init__(self,modelname) -> None:
        self.modelname = modelname

    def validing(self,valid_loader,model,criterion=None):
        """
        function:
        ---------
            验证函数，用于评估模型在验证集上的效果。

        args:
        -----
            valid_loader: 验证集数据，通过torch的DataLoader加载得到。
            model: 模型参数。
            criterion: 损失函数。
        
        returns:
        --------
            accuracy: 返回验证集上的正确率。
            f1: 返回验证集的f1分数。
            loss: 返回验证集的总体损失。

        """
        model.eval()
        predictions = []
        true_labels = []
        total_loss = 0.
        for jdx,batch in enumerate(valid_loader):
            if isinstance(model,MentalRoberta) is True:
                batchInputIds, batchInput_mask, batchLabels = batch
                batchInputIds , batchInput_mask = batchInputIds.to(CFG.device),batchInput_mask.to(CFG.device)
                batchLabels = batchLabels.to(CFG.device)
                print("batchInputIds: ",batchInputIds.shape)
                with torch.no_grad():
                    outputs = model(batchInputIds,attention_mask=batchInput_mask,labels=batchLabels,mode="val")
                    out = outputs[1]
                    out = out.detach().cpu().numpy()
                    labelIds = batchLabels.to('cpu').numpy()
                    total_loss += outputs[0].item()
            elif isinstance(model, BiLSTM) is True or isinstance(model,TextCNN) is True:
                x = batch[0].to(CFG.device).squeeze(1).long()
                y = batch[1].to(CFG.device).long()
                print("x shape: ",x.shape)
                with torch.no_grad():
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    total_loss += loss.cpu().numpy()
                    out = nn.functional.softmax(y_pred).cpu().numpy()
                    labelIds = y.cpu().numpy()
            predictions.append(out)
            true_labels.append(labelIds)
            break
            
        finalPredictions = [ele for predList in predictions for ele in predList]
        finalPredictions = np.argmax(finalPredictions, axis=1).flatten()
        finalTrueLabels = [ele for trueList in true_labels for ele in trueList]
        # 计算指标
        accuracy =  accuracy_score(finalPredictions, finalTrueLabels)
        f1 = f1_score(finalPredictions, finalTrueLabels, average = "weighted")
        print("Validation [loss: {:.3f}][accuracy: {:.3f}] [f1-score: {:.3f}]".format(total_loss/(len(valid_loader)),accuracy*100,f1))
        # print(classification_report(y_true=finalTrueLabels,y_pred=finalPredictions))
        
        return accuracy , f1 , total_loss/(len(valid_loader))


class Training(object):
    def __init__(self,model,modelname) -> None:
        self.m = model
        self.val = Validation(modelname)
        self.m_name = modelname
    
    def training(self,train_loader,valid_loader,optimizer,criterion=None,scheduler=None):
        """
        function:
        ---------
            训练函数，用于模型的训练。
        
        args:
        ---------
            train_loader: 训练数据集，通过torch的DataLoader加载.
            valid_loader: 验证数据集，通过torch的DataLoader加载.
            optimizer: 优化器，用于反向传播中的参数更新.
            criterion: 损失函数.
            scheduler: 学习率调整，在本次实验中涉及了warm_up和余弦退火算法
            
        returns:
        ----------
        train_metrics: 返回训练过程中的评价指标，包括训练损失，训练准确率和f1分数.
        valid_metrics: 返回验证过程中的评价指标，包括验证损失，验证准确率和f1分数.

        """
        train_metrics = []
        valid_metrics = []
        for i in range(CFG.EPOCH):
            predictions = []
            true_labels = []
            total_loss = 0.0
            self.m.train()
            print("Training begin... [Epoch: {:2d}] [Learning Rate: {:5f}]".format(i,optimizer.state_dict()['param_groups'][0]['lr']))
            for step, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
                # Training Roberta 
                if isinstance(self.m, MentalRoberta):
                    batch_input_ids = batch[0].to(CFG.device)
                    batch_input_mask = batch[1].to(CFG.device)
                    batch_labels = batch[2].to(CFG.device)
                    # print(batch_input_ids.shape,batch_input_mask.shape,batch_labels.shape,sep="\t")
                    outputs = self.m(x = batch_input_ids,
                                    attention_mask = batch_input_mask, 
                                    labels = batch_labels,
                                    mode = "train")
                    loss = outputs[0]
                    total_loss += loss.item()

                    out = outputs[1].cpu().detach().numpy()
                    labelIds = batch_labels.to('cpu').numpy()
                # Traning BiLSTM
                if isinstance(self.m, BiLSTM) or isinstance(self.m, TextCNN):
                    x = batch[0].to(CFG.device).squeeze(1).long()
                    y = batch[1].to(CFG.device).long()
                    y_pred = self.m(x)
                    loss = criterion(y_pred, y)
                    total_loss += loss.item()
                    
                    out = nn.functional.softmax(y_pred).cpu().detach().numpy()
                    labelIds = y.cpu().numpy()
                
                predictions.append(out)
                true_labels.append(labelIds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            if scheduler != None:
                scheduler.step()

            # 计算训练过程中的指标：loss,accuracy,f1-score
            finalPredictions = [ele for predList in predictions for ele in predList]
            finalPredictions = np.argmax(finalPredictions, axis=1).flatten()
            finalTrueLabels = [ele for trueList in true_labels for ele in trueList]
            accuracy =  accuracy_score(finalPredictions, finalTrueLabels)
            f1 = f1_score(finalPredictions, finalTrueLabels, average = "weighted")
            train_metrics.append((accuracy,f1,total_loss/len(train_loader)))
            print("Training :[EPOCH:{:2d}/{:d}] [loss:{:.3f}] [accuracy: {:.3f}] [f1-score: {:.3f}]".format(i,CFG.EPOCH,total_loss/len(train_loader),accuracy*100,f1))
            
            # 开始验证集测试
            val_acc , val_f1, val_loss =  self.val.validing(valid_loader,self.m,criterion)
            valid_metrics.append((val_acc,val_f1,val_loss))

        return train_metrics , valid_metrics

