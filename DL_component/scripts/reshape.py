import pandas as pd
import numpy as np
import argparse
import os

# python scripts/reshape.py --input_file data/owndata/attackdata/1_x_4.csv --output_file data/owndata/reshape/x_3_16.csv --group_size 16

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="通过分组行重塑CAN数据。")
    parser.add_argument("--input_file", type=str, required=True, help="输入CSV文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="保存重塑后CSV文件的路径。")
    parser.add_argument("--group_size", type=int, default=9, help="要分组的行数。")
    parser.add_argument("--discard_incomplete", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否丢弃不完整的分组。使用True或False。")
    return parser.parse_args()

def reshape_data(input_file, output_file, group_size=9, discard_incomplete=True):
    """
    通过将每 'group_size' 行分组为一行来重塑数据。
    
    参数:
    - input_file (str): 输入CSV文件路径。
    - output_file (str): 保存重塑后CSV文件的路径。
    - group_size (int): 要分组的行数。
    - discard_incomplete (bool): 是否丢弃不完整的分组。
    """
    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 读取数据
    try:
        data = pd.read_csv(input_file, header=None)
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return
    
    total_rows = data.shape[0]
    num_groups = total_rows // group_size
    remainder = total_rows % group_size
    
    if remainder != 0 and not discard_incomplete:
        num_groups += 1
    elif remainder != 0 and discard_incomplete:
        print(f"丢弃最后的 {remainder} 不完整行。")
    
    reshaped_data = []
    reshaped_labels = []
    
    for group in range(num_groups):
        start_idx = group * group_size
        end_idx = start_idx + group_size
        
        if end_idx > total_rows:
            if discard_incomplete:
                break
            else:
                end_idx = total_rows
        
        group_data = data.iloc[start_idx:end_idx]
        
        # 提取数据字段和标签
        data_fields = group_data.iloc[:, :-1].values.flatten()
        labels = group_data.iloc[:, -1].values
        
        # 聚合标签：如果任何标签为1，则设为1；否则设为0
        aggregated_label = 1 if any(labels == 1) else 0
        
        reshaped_data.append(data_fields)
        reshaped_labels.append(aggregated_label)
        
        if group % 1000 == 0 and group != 0:
            print(f"已处理 {group} 组...")
    
    # 转换为DataFrame
    reshaped_df = pd.DataFrame(reshaped_data)
    reshaped_df['label'] = reshaped_labels
    
    # 保存为CSV
    try:
        reshaped_df.to_csv(output_file, header=False, index=False)
        print(f"重塑后的数据已保存到 {output_file}")
        print(f"总共处理了 {len(reshaped_df)} 组。")
    except Exception as e:
        print(f"保存重塑后数据时出错: {e}")

def main():
    args = parse_arguments()
    reshape_data(args.input_file, args.output_file, args.group_size, args.discard_incomplete)

if __name__ == "__main__":
    main()
