import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def parse_otids_line(line):
    """
    解析OTIDS数据集文件中的单行。
    """
    match = re.search(
        r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+DLC:\s+(\d+)\s+((?:[0-9a-fA-F]{2}\s*)*)",
        line
    )
    if not match:
        return None

    timestamp = float(match.group(1))
    can_id = int(match.group(2), 16)
    data_str = match.group(4).strip()
    
    data = [int(b, 16) for b in data_str.split()] if data_str else []
    
    # 将数据填充到8个字节
    padded_data = data + [0] * (8 - len(data))
    
    # 特征将是 [ID, D0, D1, ..., D7]
    return [can_id] + padded_data

def load_all_otids_data(data_path):
    """
    加载所有OTIDS .txt文件，处理特征和标签，
    并返回两个Numpy数组。
    """
    files = {
        "Attack_free_dataset.txt": 0,
        "Impersonation_attack_dataset.txt": 1,
        "Fuzzy_attack_dataset.txt": 1,
        "DoS_attack_dataset.txt": 1
    }
    
    all_features_list = []
    all_labels_list = []

    print("正在处理OTIDS .txt文件...")
    for filename, label in files.items():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"警告：未找到文件 {file_path}")
            continue
        
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        file_features = []
        for line in tqdm(lines, desc=f"正在处理 {filename}"):
            parsed_features = parse_otids_line(line)
            if parsed_features:
                file_features.append(parsed_features)
        
        if file_features:
            all_features_list.extend(file_features)
            all_labels_list.extend([label] * len(file_features))

    if not all_features_list:
        return np.array([]), np.array([])

    all_features = np.array(all_features_list, dtype=np.float32)
    all_labels = np.array(all_labels_list, dtype=np.uint8)

    return all_features, all_labels


class OTIDSDataset(Dataset):
    """
    用于OTIDS数据集的PyTorch Dataset类。
    """
    def __init__(self, data_path, chunk_size=100, transform=None):
        """
        Args:
            data_path (str): OTIDS数据集目录的路径。
            chunk_size (int): 每个数据块的大小（例如，100个时间步）。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        features_cache_path = os.path.join(data_path, 'otids_features.npy')
        labels_cache_path = os.path.join(data_path, 'otids_labels.npy')

        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("正在从缓存加载预处理数据...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("未找到缓存，正在加载和预处理所有OTIDS数据...")
            self.all_features, self.all_labels = load_all_otids_data(data_path)
            
            if self.all_features.size > 0:
                print("正在将处理好的数据缓存到.npy文件...")
                np.save(features_cache_path, self.all_features)
                np.save(labels_cache_path, self.all_labels)
        
        if len(self.all_features) == 0:
             print("警告：未加载任何数据。")
             return

        print(f"\n数据加载完成。总时间步: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"总攻击时间步: {attack_count}")
        print(f"总正常时间步: {normal_count}")
        print(f"不平衡比率 (正常/攻击): {imbalance_ratio:.2f}")

    def __len__(self):
        if not hasattr(self, 'all_features'):
            return 0
        return len(self.all_features) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        chunk_features = self.all_features[start_idx:end_idx]
        label_slice = self.all_labels[start_idx:end_idx]
        
        label = 1 if np.any(label_slice == 1) else 0
        
        chunk = torch.from_numpy(chunk_features.astype(np.float32))
        label = torch.tensor(label).long()

        if self.transform:
            chunk = self.transform(chunk)
            
        return chunk, label

if __name__ == '__main__':
    from torch.utils.data import random_split
    
    dataset_root = './data/OTIDS'

    if not os.path.exists(dataset_root):
        print(f"数据集路径 '{dataset_root}' 未找到。")
    else:
        print("正在初始化OTIDSDataset...")
        otids_dataset = OTIDSDataset(data_path=dataset_root, chunk_size=100)
        
        if len(otids_dataset) > 0:
            dataset_size = len(otids_dataset)
            print(f"\n数据集初始化成功。总数据块数: {dataset_size}")
            
            first_chunk, first_label = otids_dataset[0]
            print(f"单个数据块的形状: {first_chunk.shape}")
            print(f"第一个数据块的标签: {first_label}")
            
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\n数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
            
            train_dataset, val_dataset, test_dataset = random_split(otids_dataset, [train_size, val_size, test_size])
            
            # 当使用多进程（num_workers > 0）时，建议在Linux上设置 'fork' 启动方法
            # import torch.multiprocessing as mp
            # mp.set_start_method('fork', force=True)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True) # num_workers > 0 may cause issues in some environments
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

            print("\nDataLoader创建成功！现在可以开始训练。")
            
            print("\n从train_loader中取一个批次进行测试...")
            try:
                for i, (chunks_batch, labels_batch) in enumerate(train_loader):
                    print(f"批次 {i+1} 数据块形状: {chunks_batch.shape}")
                    print(f"批次 {i+1} 标签形状: {labels_batch.shape}")
                    if i >= 0: # 仅测试一个批次
                        break
            except Exception as e:
                print(f"在DataLoader迭代期间发生错误: {e}")
                print("请检查您的环境。如果num_workers > 0导致问题，请尝试将其设置为0。") 