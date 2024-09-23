import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_file(file_path):
    """
    处理单个 .asc 文件，返回处理后的数据列表。
    
    参数:
        file_path (str): .asc 文件的路径。
    
    返回:
        List[List[int]]: 处理后的数据，每个子列表代表一行数据。
    """
    totaldata = []
    try:
        # 读取文件，跳过前4行，指定列名为 'col'
        data = pd.read_csv(file_path, skiprows=4, names=['col'], delimiter='\n', encoding='utf-8')
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}，错误信息: {e}")
        return totaldata
    
    # 遍历每一行数据
    for index, row in tqdm(data.iterrows(), total=len(data), desc=f"处理文件 {os.path.basename(file_path)}"):
        tempstr = row['col'].split()
        
        # 确保 tempstr 有 14 个元素，不足的用 '00' 填充
        if len(tempstr) < 14:
            tempstr += ['00'] * (14 - len(tempstr))
        
        try:
            # 解析需要的值
            x = int(tempstr[2], 16)
            payload = [int(tempstr[i], 16) for i in range(6, 14)]
            currow = [x] + payload + [0]
            totaldata.append(currow)
        except ValueError as ve:
            print(f"警告：文件 {file_path} 的第 {index + 1} 行存在无效的十六进制数，跳过该行。")
        except IndexError as ie:
            print(f"警告：文件 {file_path} 的第 {index + 1} 行字段不足，跳过该行。")
    
    return totaldata

def process_all_files(input_dir, output_dir, output_filename='merged_data.csv'):
    """
    处理输入文件夹中的所有 .asc 文件，并将结果合并保存到一个 CSV 文件中。
    
    参数:
        input_dir (str): 存放 .asc 文件的输入文件夹路径。
        output_dir (str): 处理后 CSV 文件的输出文件夹路径。
        output_filename (str): 合并后 CSV 文件的文件名。
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 .asc 文件的路径
    file_pattern = os.path.join(input_dir, '*.asc')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"警告：在目录 {input_dir} 中未找到任何 .asc 文件。")
        return
    
    print(f"找到 {len(file_list)} 个 .asc 文件。开始处理...")
    
    # 初始化总数据列表
    all_data = []
    
    # 遍历并处理每个文件
    for file_path in tqdm(file_list, desc="处理所有文件"):
        file_data = process_file(file_path)
        all_data.extend(file_data)
    
    if not all_data:
        print("警告：没有任何数据被处理和合并。")
        return
    
    # 将所有数据转换为 numpy 数组，数据类型为 int16
    temparray = np.array(all_data, dtype=np.int16)
    
    # 保存为 CSV 文件
    output_path = os.path.join(output_dir, output_filename)
    try:
        np.savetxt(output_path, temparray, fmt='%d', delimiter=',')
        print(f"所有文件的数据已成功合并并保存到 {output_path}")
    except Exception as e:
        print(f"错误：无法保存合并后的数据到 {output_path}，错误信息: {e}")
    
    # 可选：打印部分数据
    print("示例数据（前5行）：")
    print(temparray[:5])

if __name__ == "__main__":
    # 定义输入和输出目录
    input_directory = './data/owndata/origin/'      # 输入文件夹路径
    output_directory = './data/owndata/processed/' # 输出文件夹路径
    output_file = 'merged_data.csv'                 # 合并后的输出文件名
    
    process_all_files(input_directory, output_directory, output_file)
