import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

import torch
from torch.utils import data
from tqdm import tqdm
import random

from modules.model import TransformerClassifier_noposition, CNN, KNN, SVM

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="path to the saved model")
parser.add_argument("--model_type", type=str, default='SVM', help="which model will be used(KNN, CNN, Attn, SVM)")
parser.add_argument("--data_src", type=str, default='Seo', help="the dataset name")
parser.add_argument("--attack_type", type=str, default='DoS', help="which attack in: DoS, Fuzz or Gear")
parser.add_argument("--propotion", type=float, default=0.8, help="the count of train divide the count of whole")
parser.add_argument("--n_classes", type=int, default=2, help="how many classes have")
opt = parser.parse_args()
print(opt)

# 加载数据
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# 分割数据集
train_data = source_data.iloc[:datalen,:]
train_label = source_label.iloc[:datalen,:]
test_data = source_data.iloc[datalen:,:]
test_label = source_label.iloc[datalen:,:]

torch_data_test = GetDataset(test_data, test_label)
testdataloader = data.DataLoader(torch_data_test, batch_size=32, shuffle=False)

# 加载模型
if opt.model_type == 'CNN':
    model = torch.load(opt.model_path).to(device)
elif opt.model_type == 'KNN':
    model = KNeighborsClassifier()
elif opt.model_type == 'Attn':
    model = torch.load(opt.model_path).to(device)
elif opt.model_type == 'SVM':
    model = SVC(kernel='linear', C=1.0)
else:
    raise ValueError("Invalid model_type specified.")


# SVM 评估过程
if opt.model_type == 'SVM' or opt.model_type == 'KNN':
    train_data = np.array(train_data.values, dtype='float32')
    test_data = np.array(test_data.values, dtype='float32')
    train_label = np.array(train_label.values, dtype='float32').flatten()
    test_label = np.array(test_label.values, dtype='float32').flatten()

    if opt.model_type == 'KNN':
        model.fit(train_data, train_label)
    elif opt.model_type == 'SVM':
        model.fit(train_data, train_label)

    # 使用 tqdm 显示预测过程的进度条
    predictions = []
    for i in tqdm(range(len(test_data)), desc="Evaluating"):
        prediction = model.predict([test_data[i]])
        predictions.append(prediction[0])

    predictions = np.array(predictions)
    accuracy = accuracy_score(test_label, predictions)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

else:
    # 其他模型（如CNN或Transformer）的评估过程
    model.eval()
    acc, nums = 0, 0

    with torch.no_grad():
        for idx, (data_x, data_y) in enumerate(testdataloader):
            if opt.model_type == 'CNN':
                data_x = np.reshape(data_x, (32,1,9,9))

            data_x = data_x.to(torch.float32).to(device)
            data_y = data_y.to(torch.long).to(device)
            data_y = data_y.squeeze(1)

            outputs = model(data_x)
            predicts = torch.where(outputs[:,1] > 0.5, 1, 0)
            acc += sum(predicts == data_y).cpu()
            nums += data_y.size()[0]

    accuracy = acc / nums
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Attack Type: {opt.attack_type}
Final Accuracy: {accuracy}
"""

# 将评估结果写入文件
with open("eval_log.txt", "a") as f:
    f.write(output_str)

print(output_str)
