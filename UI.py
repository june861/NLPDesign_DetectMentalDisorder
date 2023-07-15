from test import testing
from config import CFG
from Model import MentalRoberta
from datapreprocess import DataPipline

from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import numpy as np
import torch
import pickle

class Args():
    def __init__(self,m) -> None:
        self.loss = "bce"
        self.m = m
        self.inplace = "local"
        self.mfile = "mental-bert-base-uncased.pth"
        self.tfile = "tokenizer-mental-bert-base-uncased.pth"
        self.glove = "./glove.6B.100d.txt"
        self.weight = 0



# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型及其分词器
roberta = torch.load("roberta-fold0-model.pth",map_location=device)["model"]
tokenizer_roberta = torch.load("tokenizer-mental-bert-base-uncased.pth")

bilstm_sys = torch.load("bilstm-fold1-model.pth",map_location=device)
bilstm = bilstm_sys["model"]
tokenizer_bilstm = bilstm_sys["tokenizer"]


text_cnn_sys = torch.load("textcnn-fold2-model.pth",map_location=device)
textcnn = text_cnn_sys["model"]
tokenizer_textcnn = text_cnn_sys["tokenizer"]

roberta.eval() 
bilstm.eval()
textcnn.eval()

# 精神疾病预测函数，返回置信度
def predict_mental_illness(title, post,model,tokenizer,mname="roberta"):
    """
    function:
    ---------
        通过输入的帖子，分别利用所指模型预测当前的精神疾病
    
    args:
    ---------
        title: 帖子的标题
        post:  帖子的内容
        model: 前端指定的模型
        tokenizer: 分词工具
        mname: 模型名称
    
    returns:
    ---------
        返回一个字典类型(Dict)的数据，形式如下所示：
        {"adhd": score1,
         "anxiety": score2,
         "bipolar": score3,
         "depression": score4,
         "ptsd": score5,
         "none": score6}

    """
    torch.cuda.empty_cache()
    mental_illnesses = ["adhd","anxiety","bipolar","depression","ptsd","none"]

    if title == "" and post == "":
        return {disease:score for disease ,score in zip(mental_illnesses,[0 for i in range(len(mental_illnesses))])}
    
    if mname == "roberta":
        data = tokenizer.encode(title + " " + post)
        data = pad_sequences([data],maxlen = min(512,len(title + " " + post)), 
                            dtype = "long", value = 0, 
                            truncating = "post", padding = "post")
        attentionMasks = [int(tokenId>0) for tokenId in data[0]]
        data ,attentionMasks = torch.tensor(data[0]), torch.tensor(attentionMasks)
        model.eval()
        with torch.no_grad():
            pred = model(data.unsqueeze(0).to(device),attentionMasks.unsqueeze(0).to(device),labels=torch.tensor([0]).to(device),mode="val")
            pred = torch.nn.functional.softmax(pred[1]).cpu().detach().numpy().tolist()[0]
            confidence_scores = pred
            print(confidence_scores)
        return {disease: score for disease, score in zip(mental_illnesses, confidence_scores)}
    
    elif mname.lower() == "bilstm" or mname.lower() == "textcnn":
        data = tokenizer.texts_to_sequences([title + " " + post])
        data = pad_sequences(data,maxlen = max(750,len(data[0])), 
                            dtype = "long", value = 0, 
                            truncating = "post", padding = "post")
        data = torch.tensor(data).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(data)
        print(pred)
        confidence_scores = torch.nn.functional.softmax(pred).cpu().detach().numpy().tolist()[0]
        print(confidence_scores)
        return {disease: score for disease, score in zip(mental_illnesses, confidence_scores)}




# 定义预测函数并设置显示样式
def predict(title, post):
    """
    functions:
    ----------
        为输入的帖子做预测
    
    args:
    ----------
        title : 帖子的标题
        post  : 帖子的内容
    
    returns:
    ----------
        返回UI界面的现实结果
    
    """

    # 每个模型在k折交叉验证下最佳的f1-score 
    loss_scores = {
        "roberta": 0.5774727364381155,
        "bilstm": 0.6382473905881246,
        "textcnn": 0.7547866751750311,
    }

    
    weights = torch.nn.functional.softmax(torch.tensor([-1 * item for _,item in loss_scores.items()]))
    model_weights = {}
    indexs = 0
    for model_name,_ in loss_scores.items():
        model_weights[model_name] = weights[indexs]
        indexs += 1

    print(model_weights)

    roberta_predictions = predict_mental_illness(title = title,post = post,model = bilstm,tokenizer = tokenizer_bilstm,mname= "bilstm")
    bilstm_predictions = predict_mental_illness(title = title,post = post,model = roberta,tokenizer = tokenizer_roberta,mname= "roberta")
    textcnn_predictions = predict_mental_illness(title = title,post = post,model = textcnn,tokenizer = tokenizer_textcnn,mname= "textcnn")

    predictions = {}
    for key in roberta_predictions.keys():
        predictions[key] = model_weights["roberta"] * roberta_predictions[key] + \
                           model_weights["bilstm"]  * bilstm_predictions[key] + \
                           model_weights["textcnn"] * textcnn_predictions[key]

    results = []
    for disease, score in predictions.items():
        bar_length = int(score * 200)   # 将预测概率映射到长度 0-200
        bar = f'<progress class="progressbar" value="{bar_length}" max="200"></progress>'
        result = f"{disease}: {score*100:.2f}% {bar}"
        results.append(result)
    return "<br>".join(results)

# 创建输入组件和输出组件
title_input = gr.inputs.Textbox(placeholder="Title",label=None)
post_input = gr.inputs.Textbox(placeholder="POST",label=None)

# 创建可选项
# options = ["BiLSTM", "RoBERTa","TextCNN"]
# option_input = gr.inputs.Radio(choices=options, label="选择一个模型")

# 创建输入和输出组件列表
inputs = [title_input, post_input]
outputs = gr.outputs.HTML()

# 创建界面
iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs,
                     title="精神疾病预测", description=None)

# 自定义界面布局和样式
iface.layout = [["标题", "内容"], ["输出"]]
iface.config["style"] = "vertical"
iface.css = """
.progressbar {
    width: 100%;
    height: 8px;
    background-color: lightgray;
    border-radius: 4px;
    overflow: hidden;
}

.progressbar::-webkit-progress-bar {
    background-color: lightgray;
}

.progressbar::-webkit-progress-value {
    background-color: red;
}

.progressbar::-moz-progress-bar {
    background-color: red;
}
"""
iface.launch(share=True)
