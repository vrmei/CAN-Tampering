import argparse
import numpy as np
import pandas as pd

import torch.utils
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import *
from AttentionModel import PositionalEncoding
from sklearn import metrics
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of traning")
parser.add_argument("--n_classes", type=int, default=2, help="how many classes have")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--lr", type=float, default=0.0001)
opt = parser.parse_args()
print(opt)

device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

class GetDataset(data.Dataset):
    def __init__(self, data_root, data_label):
        # tempdata = None
        # totaldata = None
        # first = True
        # self.data = data_root.values
        # for i in tqdm(range(len(self.data))):
        #     tempdata = np.reshape(self.data[i], (9,9))
        #     if first:
        #         totaldata = tempdata
        #         first = False
        #     else:
        #         totaldata = np.array(totaldata, tempdata)
        #         print(totaldata.shape)
        self.data = data_root.values
        self.label = data_label.values
        #return self.data, self.label
    
    def __getitem__(self, index):
        d = self.data[index]
        l = self.label[index]
        return d, l
    
    def __len__(self):
        return len(self.data)

propotion = 0.8

data = np.load("greygraph/greyimage.npy")

model = PositionalEncoding().to(device)
#criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

acc, nums = 0, 0
for epoch in range(opt.n_epochs):
    acc = nums = 0
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(traindataloader):
        try:
            data_x = np.reshape(data_x, (32,1,9,9))
        except:
            continue
        data_x = data_x.to(torch.float32).to(device)
        data_y = data_y.to(torch.float32).to(device)
        
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(data_y, outputs)
        loss.backward()
        optimizer.step()
        predicts = torch.where(outputs > 0.5, 1, 0)
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
            data_x = np.reshape(data_x, (32,1,9,9))
        except:
            continue
        data_x = data_x.to(torch.float32).to(device)
        # for i, l in enumerate(data_y):
        #     if l == 1:
        #         data_y[i] = 0
        #     else:
        #         data_y[i] = 1
        data_y = data_y.to(torch.float32).to(device)
        outputs = model(data_x)
        loss = criterion(data_y,outputs)
        test_epochs_loss.append(loss.item())
        test_loss.append(loss.item())
        predicts = torch.where(outputs > 0.5, 1, 0)
        acc += sum(predicts == data_y).cpu()
        nums += data_y.size()[0]
    
    print("epoch= {}/{}, {}/{} of test, acc=".format(
        epoch, opt.n_epochs, idx, len(testdataloader)),"%.4f" % float(acc/nums))
    #input()
    test_epochs_loss.append(np.average(test_epochs_loss))
    test_acc.append((acc/nums))


print(test_acc,"max:",max(test_acc))
torch.save(model, "SeoGearmodel.pth")
