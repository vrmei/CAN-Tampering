import os
import glob
import json
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def parse_log_line(line):
    """
    解析单行candump日志。
    日志格式: (timestamp) can_interface ID#DATA
    例如: (1602162391.566168) can0 110#02202E1300181300
    """
    match = re.match(r'\((\d+\.\d+)\)\s+\w+\s+([0-9A-F]+)#([0-9A-F]*)', line)
    if not match:
        return None, None, None
        
    timestamp = float(match.group(1))
    can_id_hex = match.group(2)
    can_id = int(can_id_hex, 16)
    
    data_hex = match.group(3).ljust(16, '0') # 填充到8字节 (16个十六进制字符)
    data = [int(data_hex[i:i+2], 16) for i in range(0, 16, 2)]
    
    return timestamp, can_id, data

def get_attack_intervals(json_path):
    """
    从JSON文件中解析攻击的开始和结束时间。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    markers = data.get('markers', [])
    starts = sorted([m['time'] for m in markers if 'Start' in m['description']])
    ends = sorted([m['time'] for m in markers if 'End' in m['description']])
    
    # 将开始和结束时间配对
    if len(starts) != len(ends) or not starts or not ends:
        print(f"警告: 在 {json_path} 中找到不匹配的开始/结束标记。")
        return []
        
    return list(zip(starts, ends))

def load_crysys_data(data_path, num_features=9):
    """
    加载所有CrySyS log文件，处理特征和标签。
    
    Args:
        data_path (str): CrySyS数据集的根目录路径。
        num_features (int): 特征数量 (1 ID + 8 Data Bytes)。
        
    Returns:
        tuple: (all_features, all_labels)
    """
    print("正在扫描log文件...")
    all_log_files = glob.glob(os.path.join(data_path, '**/*.log'), recursive=True)
    
    all_features_list = []
    all_labels_list = []

    print("正在处理log文件...")
    for log_file in tqdm(all_log_files, desc="Files"):
        is_malicious = 'malicious' in os.path.basename(log_file).lower()
        attack_intervals = []
        
        if is_malicious:
            log_basename_full = os.path.basename(log_file)
            dir_name = os.path.dirname(log_file)
            
            # 处理特殊文件名 "-inj-messages.log"
            if log_basename_full.endswith('-inj-messages.log'):
                log_basename_stem = log_basename_full.replace('-inj-messages.log', '')
            else:
                log_basename_stem = log_basename_full.replace('.log', '')

            # 查找匹配的JSON文件
            json_file_path_to_find = os.path.join(dir_name, log_basename_stem + '.json')
            potential_json_files = glob.glob(json_file_path_to_find)
            
            if potential_json_files:
                # 通常只有一个最匹配的
                json_file = potential_json_files[0]
                attack_intervals = get_attack_intervals(json_file)
            else:
                print(f"警告: 找不到与 {log_file} 对应的JSON文件 (尝试查找: {json_file_path_to_find})")
                continue

        with open(log_file, 'r') as f:
            for line in f:
                timestamp, can_id, data = parse_log_line(line.strip())
                if timestamp is None:
                    continue
                
                features = [can_id] + data
                # 确保特征维度正确
                if len(features) != num_features:
                    continue

                label = 0
                if is_malicious and attack_intervals:
                    for start, end in attack_intervals:
                        if start <= timestamp <= end:
                            label = 1
                            break
                
                all_features_list.append(features)
                all_labels_list.append(label)

    all_features = np.array(all_features_list, dtype=np.float32)
    all_labels = np.array(all_labels_list, dtype=np.uint8)

    return all_features, all_labels


class CrySySDataset(Dataset):
    """
    内存高效的CrySyS数据集PyTorch Dataset类。
    """
    def __init__(self, data_path, chunk_size=100, num_features=9, transform=None):
        """
        Args:
            data_path (str): CrySyS数据集的根目录路径。
            chunk_size (int): 每个数据块的大小。
            num_features (int): 每个时间步的特征数量 (1 ID + 8 Data).
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        features_cache_path = os.path.join(data_path, 'crysys_features.npy')
        labels_cache_path = os.path.join(data_path, 'crysys_labels.npy')

        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("正在从缓存加载预处理的CrySyS数据...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("缓存文件未找到，正在加载和预处理所有CrySyS log数据...")
            self.all_features, self.all_labels = load_crysys_data(data_path, num_features=num_features)
            
            print("正在缓存处理好的数据为.npy文件...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        
        if len(self.all_features) == 0:
            print("错误：未能加载任何数据。请检查路径和文件格式。")
            return

        print(f"\nCrySyS数据加载完成。总时间步: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"总攻击时间步: {attack_count}, 总正常时间步: {normal_count}")
        print(f"不平衡比率 (正常/攻击): {imbalance_ratio:.2f}")

    def __len__(self):
        return len(self.all_features) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        chunk_features = self.all_features[start_idx:end_idx]
        label_slice = self.all_labels[start_idx:end_idx]
        
        label = 1 if np.any(label_slice == 1) else 0
        
        chunk = torch.from_numpy(chunk_features)
        label = torch.tensor(label).long()

        if self.transform:
            chunk = self.transform(chunk)
            
        return chunk, label

if __name__ == '__main__':
    from torch.utils.data import random_split
    
    dataset_root = './MIDS/data/CrySyS'

    if not os.path.exists(dataset_root):
        print(f"数据集路径 '{dataset_root}' 未找到。")
    else:
        print("正在初始化CrySySDataset...")
        # num_features = 1 ID + 8 data bytes
        crysys_dataset = CrySySDataset(data_path=dataset_root, chunk_size=100, num_features=9)
        
        if len(crysys_dataset) > 0:
            dataset_size = len(crysys_dataset)
            print(f"\n数据集初始化成功。可生成的总数据块数: {dataset_size}")
            
            first_chunk, first_label = crysys_dataset[0]
            print(f"单个数据块的形状: {first_chunk.shape}")
            print(f"第一个数据块的标签: {first_label}")
            
            # 划分为训练、验证和测试集
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\n数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
            
            train_dataset, val_dataset, test_dataset = random_split(crysys_dataset, [train_size, val_size, test_size])
            
            # 创建DataLoader
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print("\nDataLoader创建成功！现在可以开始训练。")
            
            print("\n从train_loader中取一个批次进行测试...")
            # 使用 try-except 块来处理 num_workers 在某些系统上的问题
            try:
                for chunks_batch, labels_batch in train_loader:
                    print(f"一个批次数据块的形状: {chunks_batch.shape}") # 应该是 [64, 100, 9]
                    print(f"一个批次标签的形状: {labels_batch.shape}")   # 应该是 [64]
                    break
            except Exception as e:
                print(f"在测试DataLoader时发生错误: {e}")
                print("尝试不使用多线程工作进程 (num_workers=0)...")
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
                for chunks_batch, labels_batch in train_loader:
                    print(f"一个批次数据块的形状 (无多线程): {chunks_batch.shape}")
                    print(f"一个批次标签的形状 (无多线程): {labels_batch.shape}")
                    break

        else:
            print("未能生成任何数据块。请检查数据路径和文件。") 