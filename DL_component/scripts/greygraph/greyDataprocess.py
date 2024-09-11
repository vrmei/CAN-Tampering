import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import torch
import cv2
from tqdm import *

data = pd.read_csv('greygraph/whitetest_1000ms.asc')
data = data.iloc[3:]
data.columns = ['col']

data2 = pd.read_csv('greygraph/whitetest.asc')
data2 = data2.iloc[3:]
data2.columns = ['col']

x_data = range(0,2048,1)
total_y_data = []
total_y_data2 = []

first = True

timestart = 0

counter_code = list()
totaldata = []
totalcnt = []
nowtime = []

counter_code2 = list()
totaldata2 = []
totalcnt2 = []
nowtime2 = []

for index, row in tqdm(data.iterrows()):
    #print(type(row['col']))😊😊
    
    tempstr = row['col'].split()
    if first:
        timestart = tempstr[0]
        first = False
    while len(tempstr) != 14:
        tempstr.append('00')

    x = int(tempstr[2], 16)
    if float(tempstr[0]) - float(timestart) < 1:
        counter_code.append(x)
    else:
        nowtime.append(tempstr[0])
        timestart = tempstr[0]
        cntcode = Counter(counter_code)
        totalcnt.append(cntcode)
        totaldata.append(counter_code)
        counter_code = []
        #print(tempstr)

for cnt in totalcnt:
    y_data = [0] * len(x_data)
    for key, value in cnt.items():
        y_data[key] = value
    total_y_data.append(y_data)
    


first = True
tempstart = 0
for index, row in tqdm(data2.iterrows()):
    #print(type(row['col']))😊😊
    
    tempstr = row['col'].split()
    if first:
        timestart = tempstr[0]
        first = False
    while len(tempstr) != 14:
        tempstr.append('00')

    x = int(tempstr[2], 16)
    if float(tempstr[0]) - float(timestart) < 1:
        counter_code2.append(x)
    else:
        nowtime.append(tempstr[0])
        timestart = tempstr[0]
        cntcode = Counter(counter_code2)
        totalcnt2.append(cntcode)
        totaldata2.append(counter_code2)
        counter_code2 = []
        #print(tempstr)

for cnt in totalcnt2:
    y_data = [0] * len(x_data)
    for key, value in cnt.items():
        y_data[key] = value
    total_y_data2.append(y_data)
temparray = np.array(total_y_data, dtype= np.uint8)
temparray = np.delete(temparray, np.where(~temparray.any(axis=0))[0], axis=1)

temparray2 = np.array(total_y_data2, dtype= np.uint8)
temparray2 = np.delete(temparray2, np.where(~temparray2.any(axis=0))[0], axis=1)

maxarrarylen1 = 0

maxarrarylen2 = 0
for i in range(len(temparray)):
    maxarrarylen1 = max(maxarrarylen1, len(temparray[i]))
for i in range(len(temparray2)):
    maxarrarylen2 = max(maxarrarylen2, len(temparray2[i]))

x_data = range(max(maxarrarylen1, maxarrarylen2))       
#cv2.imshow("test",temparray)

fileobject = open("greygraph/numpydata",'w')
for data in totaldata:
    fileobject.write(str(data))
    fileobject.write('\n')
    
fileobject.close()

input()
for time, y_data in enumerate(temparray):
    y_data2 = temparray2[time]
    plt.title("Now Time:"+nowtime[time]+"s")
    plt.plot(x_data, y_data)
    #plt.plot(x_data, y_data2)
    plt.pause(1)
    plt.cla()