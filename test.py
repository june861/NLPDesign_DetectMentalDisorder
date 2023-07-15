from Model import MentalRoberta,BiLSTM,TextCNN
from config import CFG

import numpy as np
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch
import warnings
import os
warnings.filterwarnings("ignore")

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score , classification_report ,confusion_matrix

np.random.seed(CFG.seed_val)
torch.manual_seed(CFG.seed_val)
torch.cuda.manual_seed_all(CFG.seed_val)

def testing(test_loader,model,args,kfold,criterion=None):
    """
    function:
    ---------
        测试函数，用于评估模型在测试集上的表现，同时打印分类报告.
    
    args:
    ---------
        test_loader: 测试集数据，经过torch的DataLoader加载的数据.
        model: 要测试的模型
        args: 系统参数
        kfold: 当前的迭代次序
        criterion: 损失函数
    
    returns:
    ---------
        accuracy: 返回测试集上的正确率。
        f1: 返回测试集的f1分数。
        loss: 返回测试集的总体损失。
    
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.
    # confuse_matrixs = [[0 for i in range(CFG.num_classes)] for j in range(CFG.num_classes)]
    print("Testing Start...")
    for jdx,batch in tqdm(enumerate(test_loader),total=len(test_loader)):
        if isinstance(model,MentalRoberta) is True:
            batchInputIds, batchInput_mask, batchLabels = batch
            batchInputIds , batchInput_mask = batchInputIds.to(CFG.device),batchInput_mask.to(CFG.device)
            batchLabels = batchLabels.to(CFG.device)
            with torch.no_grad():
                outputs = model(batchInputIds,attention_mask=batchInput_mask,labels=batchLabels,mode="val")
                out = outputs[1]
                out = out.detach().cpu().numpy()
                labelIds = batchLabels.to('cpu').numpy()
                total_loss += outputs[0].item()
        elif isinstance(model, BiLSTM) is True or isinstance(model, TextCNN) is True:
            x = batch[0].to(CFG.device).squeeze(1).long()
            y = batch[1].to(CFG.device).long()
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
                out = nn.functional.softmax(y_pred).cpu().numpy()
                labelIds = y.cpu().numpy()
        predictions.append(out)
        true_labels.append(labelIds)
    finalPredictions = [ele for predList in predictions for ele in predList]
    finalPredictions = np.argmax(finalPredictions, axis=1).flatten()
    finalTrueLabels = [ele for trueList in true_labels for ele in trueList]
    # 计算指标
    accuracy =  accuracy_score(finalPredictions, finalTrueLabels)
    f1 = f1_score(finalPredictions, finalTrueLabels, average = "weighted")
    print("Testing: [accuracy: {:.3f}] [f1-score: {:.3f}]".format(accuracy*100,f1))
    print(classification_report(y_true=finalTrueLabels,y_pred=finalPredictions))
    # 计算混淆矩阵
    cm = confusion_matrix(y_true=finalTrueLabels,y_pred=finalPredictions)
    # 保存混淆矩阵图
    cm = pd.DataFrame(cm,columns=CFG.classes,index=CFG.classes)
    hotmap = sns.heatmap(cm,cmap="YlGnBu_r",fmt="d",annot=True)
    hotmap.get_figure().savefig(os.path.join(CFG.outputdir,f"confuse-matrix-{args.m}-fold{kfold}.png"),dpi = 400)

    return accuracy , f1 , total_loss/(len(test_loader))

 