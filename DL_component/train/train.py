import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import pandas as pd

import torch.utils
from torch.utils import data
from torchvision import datasets


import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import *
from modules.model import TransformerClassifier_noposition, CNN, KNN
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", type=str, default='CE', help="the loss func(MSE, CE)")
parser.add_argument("--model_type", type=str, default='KNN', help="which model will be used(KNN, CNN, Attn, SVM)")
parser.add_argument("--data_src", type=str, default='Seo', help="the dataset where generated")
parser.add_argument("--attack_type", type=str, default='DoS', help="which attack in: DoS, Fuzz or Gear")
parser.add_argument("--propotion", type=float, default=0.8, help="the count of train divide the count of whole")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of traning")
parser.add_argument("--n_classes", type=int, default=2, help="how many classes have")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--lr", type=float, default=0.0001)
opt = parser.parse_args()
print(opt)

if opt.data_src == 'Seo':
    if opt.attack_type == 'DoS': 
        source_data = pd.read_csv('data/CNN_data/DoS_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/DoS_label.csv')

    elif opt.attack_type == 'Fuzz': 
        source_data = pd.read_csv('data/CNN_data/Fuzz_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/Fuzz_label.csv')

    elif opt.attack_type == 'gear': 
        source_data = pd.read_csv('data/CNN_data/gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/gear_label.csv')

elif opt.data_src == 'own':
    if opt.attack_type == 'DoS': 
        source_data = pd.read_csv('gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('gear_label.csv')

    elif opt.attack_type == 'Fuzz': 
        source_data = pd.read_csv('gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('gear_label.csv')

    elif opt.attack_type == 'gear': 
        source_data = pd.read_csv('gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('gear_label.csv')

device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

class GetDataset(data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root.values
        self.label = data_label.values

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
    
    def __len__(self):
        return len(self.data)

train_data = source_data.iloc[:datalen,:]
train_label = source_label.iloc[:datalen,:]
test_data = source_data.iloc[datalen:,:]
test_label = source_label.iloc[datalen:,:]

templist = []


torch_data_train = GetDataset(train_data, train_label)
torch_data_test = GetDataset(test_data, test_label)
# for i in range(torch_data.__len__()):
#     print(torch_data.__getitem__(i)[0].shape)
#     tempdata = np.reshape(torch_data.__getitem__(i)[0],(9,9))
#     print(tempdata)

traindataloader = data.DataLoader(torch_data_train, batch_size=32, shuffle=True)
#data_train = Dataloader("preprocess1.csv",)
testdataloader = data.DataLoader(torch_data_test, batch_size=32, shuffle=True)

if opt.model_type == 'CNN':
    model = CNN().to(device)
elif opt.model_type == 'KNN':
    model = KNN()
elif opt.model_type == 'Attn':
    model = TransformerClassifier_noposition(input_dim=81, num_heads=8, num_layers=2, hidden_dim=32, num_classes=2).to(device)


output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Attack Type: {opt.attack_type}
Training Proportion: {opt.propotion}
Number of Classes: {opt.n_classes}
"""
print(output_str)

output_str += '\n\n'

if opt.model_type == 'KNN':
    # 如果选择KNN，直接执行KNN的fit和predict过程
    print("Training KNN model...")
    model.fit(train_data.values, train_label.values.ravel())  # KNN的训练
    
    print("Predicting with KNN model...")
    
    # 使用 tqdm 显示预测过程的进度
    predictions = []
    for x in tqdm(test_data.values, desc="KNN Predictions"):
        predictions.append(model._predict_single(x))  # 预测每个样本
    
    predictions = np.array(predictions)  # 将结果转换为 NumPy 数组
    acc = np.mean(predictions == test_label.values.ravel())  # 计算准确率
    
    # 输出进度和最终结果
    output_str += f'Final Accuracy: {acc}\n'
    with open("log.txt", "a") as f:
        f.write(output_str)
    print(f"Final Accuracy of KNN: {acc}")
    exit()

if opt.loss_type == 'MSE':
    criterion = torch.nn.MSELoss()
elif opt.loss_type == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

test_acc = []
train_loss = []
test_loss = []
train_epochs_loss = []
test_epochs_loss = []
acc, nums = 0, 0
for epoch in range(opt.n_epochs):
    acc = nums = 0
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(traindataloader):
        try:
            if opt.model_type == 'CNN':
                data_x = np.reshape(data_x, (32,1,9,9))
        except:
            continue

        if opt.loss_type == 'CE':
            data_x = data_x.to(torch.float32).to(device)
        else:
            data_x = data_x.to(torch.long).to(device)
        data_y = data_y.to(torch.long).to(device)
        data_y = data_y.squeeze(1)
        
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()
        predicts = torch.where(outputs[:,1] > 0.5, 1, 0)
        acc += sum(predicts == data_y).cpu()
        nums += data_y.size()[0]
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx%(len(traindataloader)//100) == 0:
            print("epoch= {}/{}, {}/{} of train, loss={}".format(
                epoch, opt.n_epochs, idx, len(traindataloader),loss.item()))
    print("ACC:",100 * acc / nums)
    train_epochs_loss.append(np.average(train_epoch_loss))
    acc = nums = 0

    for idx, (data_x, data_y) in enumerate(testdataloader):
        try:
            if opt.model_type == 'CNN':
                data_x = np.reshape(data_x, (32,1,9,9))
        except:
            continue
        if opt.loss_type == 'CE':
            data_x = data_x.to(torch.float32).to(device)
        else:
            data_x = data_x.to(torch.long).to(device)
        data_y = data_y.to(torch.long).to(device)
        data_y = data_y.squeeze(1)

        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        test_epochs_loss.append(loss.item())
        test_loss.append(loss.item())
        predicts = torch.where(outputs[:,1] > 0.5, 1, 0)
        acc += sum(predicts == data_y).cpu()
        nums += data_y.size()[0]
    
    print("epoch= {}/{}, {}/{} of test, acc=".format(
        epoch, opt.n_epochs, idx, len(testdataloader)),"%.4f" % float(acc/nums))
    #input()
    test_epochs_loss.append(np.average(test_epochs_loss))
    test_acc.append((acc/nums))

output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Attack Type: {opt.attack_type}
Training Proportion: {opt.propotion}
Number of Epochs: {opt.n_epochs}
Number of Classes: {opt.n_classes}
Latent Dimension: {opt.latent_dim}
Learning Rate: {opt.lr}

Final Accuracy: {test_acc}
Max Accuracy: {max(test_acc)}
Mean Accuracy: {sum(test_acc) / len(test_acc)}
"""

with open("log.txt", "a") as f:
    f.write(output_str)

print(test_acc,"max:",max(test_acc), "   mean:", sum(test_acc) / len(test_acc))
torch.save(model, "./output/" + opt.model_type + opt.data_src + "model.pth")
