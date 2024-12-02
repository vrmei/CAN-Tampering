import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


# python scripts/own_data_process.py batch --input_dir ./data/owndata/origin/high-speed --output_dir ./data/owndata/processed/ --output_file high-speed-merge.csv
# python scripts/own_data_process.py single --file ./data/owndata/origin/high-speed/high-speed4.asc --output_dir ./data/owndata/processed/high-speed
# python scripts/own_data_process.py batch_single --input_dir ./data/owndata/origin/high-speed --output_dir ./data/owndata/processed/high-speed

def process_files_individually(input_dir, output_dir):
    """
    按单文件的方式逐一处理目录中的所有 .asc 文件。
    
    参数:
        input_dir (str): 存放 .asc 文件的输入文件夹路径。
        output_dir (str): 处理后 CSV 文件的输出文件夹路径。
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 .asc 文件的路径
    file_pattern = os.path.join(input_dir, '*.asc')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"警告：在目录 {input_dir} 中未找到任何 .asc 文件。")
        return
    
    print(f"找到 {len(file_list)} 个 .asc 文件。开始逐文件处理...")
    
    # 遍历并处理每个文件
    for file_path in tqdm(file_list, desc="Processing files individually"):
        # 提取文件名生成对应的输出文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_processed.csv"
        
        # 调用单文件处理方法
        process_single_file(file_path, output_dir, output_filename)
    
    print(f"所有文件已按单文件方式处理并保存到目录 {output_dir}")

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
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过前4行
            for _ in range(4):
                next(f)
            
            # 逐行读取文件内容
            for line_number, line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file_path)}"), 1):
                tempstr = line.strip().split()
                
                # 确保 tempstr 有 14 个元素，不足的用 '00' 填充
                if len(tempstr) < 14:
                    tempstr += ['00'] * (14 - len(tempstr))
                
                try:
                    # 解析需要的值
                    x = int(tempstr[2], 16)
                    payload = [int(tempstr[i], 16) for i in range(6, 14)]
                    currow = [x] + payload + [0]
                    totaldata.append(currow)
                except ValueError:
                    print(f"警告：文件 {file_path} 的第 {line_number + 4} 行存在无效的十六进制数，跳过该行。")
                except IndexError:
                    print(f"警告：文件 {file_path} 的第 {line_number + 4} 行字段不足，跳过该行。")
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}，错误信息: {e}")
    
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
    for file_path in tqdm(file_list, desc="Processing all files"):
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
    
    # 打印部分数据
    print("示例数据（前5行）：")
    print(temparray[:5])

def process_single_file(file_path, output_dir, output_filename=None):
    """
    处理单个 .asc 文件，并将结果保存到 CSV 文件中。
    
    参数:
        file_path (str): .asc 文件的路径。
        output_dir (str): 处理后 CSV 文件的输出文件夹路径。
        output_filename (str, optional): 输出 CSV 文件的文件名。如果未提供，将基于输入文件名生成。
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理文件
    print(f"开始处理单个文件: {file_path}")
    file_data = process_file(file_path)
    
    if not file_data:
        print("警告：没有任何数据被处理。")
        return
    
    # 将数据转换为 numpy 数组，数据类型为 int16
    temparray = np.array(file_data, dtype=np.int16)
    
    # 生成输出文件名
    if not output_filename:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_processed.csv"
    
    # 保存为 CSV 文件
    output_path = os.path.join(output_dir, output_filename)
    try:
        np.savetxt(output_path, temparray, fmt='%d', delimiter=',')
        print(f"文件的数据已成功保存到 {output_path}")
    except Exception as e:
        print(f"错误：无法保存数据到 {output_path}，错误信息: {e}")
    
    # 可选：打印部分数据
    print("示例数据（前5行）：")
    print(temparray[:5])

def main():
    parser = argparse.ArgumentParser(description="处理 .asc 文件并转换为 CSV 格式。")
    subparsers = parser.add_subparsers(dest='command', help='子命令帮助')

    # 子命令：批量处理（合并输出）
    parser_batch = subparsers.add_parser('batch', help='批量处理多个 .asc 文件并合并为一个输出文件')
    parser_batch.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径，包含 .asc 文件')
    parser_batch.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser_batch.add_argument('--output_file', type=str, default='merged_data.csv', help='合并后的输出 CSV 文件名')

    # 子命令：单文件处理
    parser_single = subparsers.add_parser('single', help='处理单个 .asc 文件')
    parser_single.add_argument('--file', type=str, required=True, help='要处理的 .asc 文件路径')
    parser_single.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser_single.add_argument('--output_file', type=str, default=None, help='输出 CSV 文件名（可选）')

    # 子命令：逐文件单独处理
    parser_individual = subparsers.add_parser('batch_single', help='按单文件方式逐一处理目录中的所有文件')
    parser_individual.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径，包含 .asc 文件')
    parser_individual.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')

    args = parser.parse_args()

    if args.command == 'batch':
        process_all_files(args.input_dir, args.output_dir, args.output_file)
    elif args.command == 'single':
        process_single_file(args.file, args.output_dir, args.output_file)
    elif args.command == 'batch_single':
        process_files_individually(args.input_dir, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
