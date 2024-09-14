import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Load data, skipping the first 3 rows and renaming the column to 'col'
filename = 'white.asc'
data = pd.read_csv(f'data/owndata/origin/{filename}', skiprows=4, names=['col'])

total_y_data = []
totaldata = []
totalcnt = []
nowtime = []
counter_code = []

# Set x_data as a range object
x_data = range(2048)

# Initialize timestart outside the loop
timestart = None

# Iterate through the data with tqdm for progress tracking
for index, row in tqdm(data.iterrows(), total=len(data)):
    tempstr = row['col'].split()

    # Set the timestart value during the first iteration
    if timestart is None:
        timestart = tempstr[0]

    # Ensure tempstr has 14 elements by appending '00' where necessary
    tempstr += ['00'] * (14 - len(tempstr))

    # Parse the necessary values
    x = int(tempstr[2], 16)
    payload = [int(tempstr[i], 16) for i in range(6, 13)]
    currow = [x] + payload
    currow += '0'
    totaldata.append(currow)

# Convert totalcnt data to y_data
for cnt in totalcnt:
    y_data = [cnt.get(i, 0) for i in x_data]
    total_y_data.append(y_data)

# Convert totaldata to a numpy array of dtype int16
temparray = np.array(totaldata, dtype=np.int16)

np.savetxt(f'./data/owndata/processed/{filename[:-4]}.csv', temparray, fmt='%d')
print(temparray)