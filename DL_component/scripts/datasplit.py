import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import torch
import cv2
from tqdm import *

data = pd.read_csv(r'C:\Users\iie\Desktop\677\CANFUZZ\py\own\data\owndata\white1.asc')
data = data.iloc[3:]
data.columns = ['col']

total_y_data = []
first = True

counter_code = list()
totaldata = []
totalcnt = []
nowtime = []
x_data = range(0,2048,1)
for index, row in tqdm(data.iterrows()):
    #print(row['col'])
    
    tempstr = row['col'].split()
    if first:
        timestart = tempstr[0]
        first = False
    while len(tempstr) != 14:
        tempstr.append('00')

    x = int(tempstr[2], 16)
    payload = [int(tempstr[4], 16), int(tempstr[5], 16), int(tempstr[6], 16), int(tempstr[7], 16), int(tempstr[8], 16), int(tempstr[9], 16), int(tempstr[10], 16), int(tempstr[11], 16)]
    currow = [x] + payload
    totaldata.append(currow)
    print(currow)
    if float(tempstr[0]) - float(timestart) < 1:
        counter_code.append(x)
    else:
        nowtime.append(tempstr[0])
        timestart = tempstr[0]
        cntcode = Counter(counter_code)
        totalcnt.append(cntcode)
        counter_code = []
        #print(tempstr)
        
for cnt in totalcnt:
    y_data = [0] * len(x_data)
    for key, value in cnt.items():
        y_data[key] = value
    total_y_data.append(y_data)
    
temparray = np.array(totaldata, dtype= np.int16)

print(temparray)