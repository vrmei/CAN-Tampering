import numpy as np
import pandas as pd

def csv_to_npy(input_csv, output_npy):
    """
    Convert a CSV file to a NumPy array and save it as an .npy file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_npy (str): Path to save the output .npy file.
    """
    try:
        # 读取 CSV 文件为 Pandas DataFrame
        data = pd.read_csv(input_csv)
        
        # 转换为 NumPy 数组
        data_array = data.values
        
        # 保存为 .npy 文件
        np.save(output_npy, data_array)
        print(f"Successfully converted {input_csv} to {output_npy}")
    except Exception as e:
        print(f"Error converting {input_csv} to .npy: {e}")

# 示例用法
if __name__ == "__main__":
    input_csv_path = "data/owndata/merged/all.csv"  # 替换为 CSV 文件路径
    output_npy_path = "data/owndata/merged/all.npy"  # 替换为保存的 .npy 文件路径
    csv_to_npy(input_csv_path, output_npy_path)
