import pandas as pd
import os
import random
from tqdm import tqdm

# 文件路径
file_path = 'data/owndata/origin/standby/standby9.asc'

# 存储分割文件的路径
output_dir = 'data/owndata/split/standby9/'
os.makedirs(output_dir, exist_ok=True)

# 存储数据的列表
data = []

# 解析数据
with open(file_path, 'r') as file:
    for line in file:
        if line.strip() and line.lstrip()[0].isdigit():
            # 拆分行内容为列表，并加入到 data 列表
            data.append(line.split())

# 转换为 DataFrame
columns = ['Timestamp'] + ['Col' + str(i) for i in range(1, len(data[0]))]
df = pd.DataFrame(data, columns=columns)

# 替换 "None" 为 ""
df.replace(to_replace="None", value="", inplace=True)

# 转换时间戳为浮点数
df['Timestamp'] = df['Timestamp'].astype(float)

# 计算时间跨度
time_min = df['Timestamp'].min()
time_max = df['Timestamp'].max()
time_span = time_max - time_min

# 确定分割时间间隔
interval = time_span / 9

# 初始化时间偏移量
time_offset = 0

# 创建分割文件
for i in range(9):
    start_time = time_min + i * interval
    end_time = start_time + interval
    split_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] < end_time)]

    # 为最后一部分包含最大时间戳
    if i == 8:
        split_df = df[df['Timestamp'] >= start_time]

    # 更新时间戳：加上偏移量
    split_df['Timestamp'] += time_offset

    # 随机生成新的时间偏移量（适用于下一个文件）
    if i < 8:  # 最后一个文件无需生成偏移量
        time_offset += random.randint(10000, 200000) / 100

    # 保存分割文件
    output_file = os.path.join(output_dir, f'low-speed1_part_{i + 1}.asc')
    with open(output_file, 'w') as out_file:
        # 写入时间偏移量到文件的第一行
        out_file.write(f"# Time Offset: {time_offset:.2f}\n")
        # 写入数据行，处理小数点后位数和 None
        for _, row in tqdm(split_df.iterrows()):
            row_values = [
                f"{float(row['Timestamp']):.6f}"  # 时间戳保留小数点后 6 位
            ] + [
                str(v) if v != "None" else "" for v in row.values[1:]
            ]
            out_file.write(" ".join(row_values) + "\n")

print(f"文件已分割为 20 部分，保存到目录：{output_dir}")
