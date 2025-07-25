import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def prepare_road_data(data_path, window_size=100, num_features=23):
    """
    加载并处理来自signal_extractions文件夹的ROAD数据集。
    
    Args:
        data_path (str): ROAD数据集的根目录路径 (例如, './data/road').
        window_size (int): 用于创建序列的滑动窗口大小。
        num_features (int): 每个时间步的特征数量 (ID + Signals)。

    Returns:
        tuple: 一个包含两个numpy数组的元组 (sequences, labels)。
    """
    ambient_files = glob.glob(os.path.join(data_path, 'signal_extractions/ambient/*.csv'))
    attack_files = glob.glob(os.path.join(data_path, 'signal_extractions/attacks/*.csv'))
    
    all_files = ambient_files + attack_files
    
    all_features_list = []
    all_labels_list = []

    print("正在处理CSV文件...")
    for file_path in tqdm(all_files, desc="Files"):
        df = pd.read_csv(file_path)

        # 处理 'accelerator' 攻击的特殊情况
        if 'accelerator' in os.path.basename(file_path).lower():
            df['Label'] = 1
        
        # 提取标签
        labels = df['Label'].values
        
        # 提取特征 (ID 和 Signals)
        # 删除非特征列
        feature_df = df.drop(columns=['Label', 'Time'])
        
        # 处理NaN - 使用0填充是一种常用策略
        feature_df = feature_df.fillna(0)
        
        # 确保特征数量正确 (num_features = 1 ID + 22 Signals)
        # 第一列是ID，其余是信号
        id_col = feature_df.iloc[:, 0].values.reshape(-1, 1)
        signal_cols = feature_df.iloc[:, 1:]

        num_signals_available = signal_cols.shape[1]
        num_signals_needed = num_features - 1

        if num_signals_available < num_signals_needed:
            # 如果信号列不够，用0填充
            padding = np.zeros((signal_cols.shape[0], num_signals_needed - num_signals_available))
            signals = np.hstack([signal_cols.values, padding])
        else:
            # 如果信号列太多，则截断
            signals = signal_cols.iloc[:, :num_signals_needed].values
            
        features = np.hstack([id_col, signals])
        
        all_features_list.append(features)
        all_labels_list.append(labels)

    # 合并所有文件的数据
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    # 使用滑动窗口创建序列
    print("正在使用滑动窗口创建序列...")
    sequences = []
    sequence_labels = []
    
    for i in tqdm(range(len(all_features) - window_size + 1), desc="Sequences"):
        sequences.append(all_features[i:i+window_size])
        
        window_labels = all_labels[i:i+window_size]
        # 如果窗口中的任何消息是攻击，则整个序列就是攻击
        sequence_label = 1 if np.any(window_labels == 1) else 0
        sequence_labels.append(sequence_label)
        
    return np.array(sequences), np.array(sequence_labels)


class ROADDataset(Dataset):
    """
    用于ROAD（基于信号）数据集的PyTorch Dataset类。
    """
    def __init__(self, data_path, window_size=100, num_features=23, transform=None):
        """
        Args:
            data_path (str): ROAD数据集的根目录路径。
            window_size (int): 滑动窗口的大小。
            num_features (int): 每个时间步的特征数量。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.sequences, self.labels = prepare_road_data(data_path, window_size, num_features)
        self.transform = transform
        print(f"数据集初始化完成。共找到 {len(self.sequences)} 个序列。")
        attack_count = np.sum(self.labels)
        normal_count = len(self.labels) - attack_count
        print(f"攻击样本数量: {attack_count}, 正常样本数量: {normal_count}")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 转换为torch张量
        sequence = torch.from_numpy(sequence).float()
        label = torch.tensor(label).long()

        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, label

if __name__ == '__main__':
    from torch.utils.data import random_split
    
    # 使用示例：
    # 假设数据集位于 './data/road'
    dataset_root = './data/road'

    # 检查路径是否存在
    if not os.path.exists(dataset_root):
        print(f"数据集路径 '{dataset_root}' 未找到。")
        print("请确保ROAD数据集已下载并解压到正确位置。")
    else:
        # 创建数据集对象
        print("正在初始化ROADDataset...")
        # 您的模型现在需要23个特征
        road_dataset = ROADDataset(data_path=dataset_root, window_size=100, num_features=23)
        
        # 您现在可以将其与DataLoader一起使用
        if len(road_dataset) > 0:
            first_sequence, first_label = road_dataset[0]
            print(f"\n单个序列的形状: {first_sequence.shape}")
            print(f"第一个序列的标签: {first_label}")
            
            # 划分为训练、验证和测试集的示例 (例如, 70-15-15的比例)
            dataset_size = len(road_dataset)
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\n数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
            train_dataset, val_dataset, test_dataset = random_split(road_dataset, [train_size, val_size, test_size])
            
            # 为每个集合创建DataLoader
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            print("\nDataLoader创建成功！")
            # 从训练加载器中获取一个批次
            for sequences_batch, labels_batch in train_loader:
                print(f"一个批次序列的形状: {sequences_batch.shape}")
                print(f"一个批次标签的形状: {labels_batch.shape}")
                # 您现在可以在训练和评估循环中使用这些加载器。
                break
        else:
            print("未能生成任何序列。请检查数据路径和文件。") 