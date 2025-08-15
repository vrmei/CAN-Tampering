import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_all_data(data_path, num_signals_needed=22):
    """
    加载所有CSV文件，处理特征和标签，返回两个Numpy数组。
    
    Args:
        data_path (str): ROAD数据集的根目录路径。
        num_signals_needed (int): 需要的信号特征数量。

    Returns:
        tuple: (all_features, all_labels)
    """
    ambient_files = glob.glob(os.path.join(data_path, 'signal_extractions/ambient/*.csv'))
    attack_files = glob.glob(os.path.join(data_path, 'signal_extractions/attacks/*.csv'))
    
    all_files = ambient_files + attack_files
    
    all_features_list = []
    all_labels_list = []

    print("正在处理CSV文件...")
    for file_path in tqdm(all_files, desc="Files"):
        df = pd.read_csv(file_path)

        if 'accelerator' in os.path.basename(file_path).lower():
            df['Label'] = 1
        
        labels = df['Label'].values
        feature_df = df.drop(columns=['Label', 'Time'])
        feature_df = feature_df.fillna(0)
        
        id_col = feature_df.iloc[:, 0].values.reshape(-1, 1)
        signal_cols = feature_df.iloc[:, 1:].values

        num_signals_available = signal_cols.shape[1]

        if num_signals_available < num_signals_needed:
            padding = np.zeros((signal_cols.shape[0], num_signals_needed - num_signals_available))
            signals = np.hstack([signal_cols, padding])
        else:
            signals = signal_cols[:, :num_signals_needed]
            
        features = np.hstack([id_col, signals])
        
        all_features_list.append(features)
        all_labels_list.append(labels)

    all_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels_list, axis=0).astype(np.uint8)

    return all_features, all_labels


class ROADDataset(Dataset):
    """
    内存高效的ROAD数据集PyTorch Dataset类。
    在 __getitem__ 中动态创建非重叠的数据块。
    """
    def __init__(self, data_path, chunk_size=100, num_features=23, transform=None):
        """
        Args:
            data_path (str): ROAD数据集的根目录路径。
            chunk_size (int): 每个数据块的大小 (例如, 100个时间步)。
            num_features (int): 每个时间步的特征数量 (1 ID + 22 Signals)。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        # 定义缓存文件的路径
        features_cache_path = os.path.join(data_path, 'all_features.npy')
        labels_cache_path = os.path.join(data_path, 'all_labels.npy')

        # 检查缓存文件是否存在
        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("正在从缓存加载预处理数据...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("缓存文件未找到，正在加载和预处理所有CSV数据...")
            self.all_features, self.all_labels = load_all_data(data_path, num_signals_needed=num_features - 1)
            
            # 缓存处理好的数据为.npy文件
            print("正在缓存处理好的数据为.npy文件...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        
        print(f"\n数据加载完成。总时间步: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        
        # 计算样本不平衡比率
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"总攻击时间步: {attack_count}, 总正常时间步: {normal_count}")
        print(f"不平衡比率 (正常/攻击): {imbalance_ratio:.2f}")

    def __len__(self):
        # 样本的总数是总长度除以数据块大小
        return len(self.all_features) // self.chunk_size

    def __getitem__(self, idx):
        # 计算非重叠块的起始和结束索引
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # 动态地从完整数据中切片出数据块
        chunk_features = self.all_features[start_idx:end_idx]
        label_slice = self.all_labels[start_idx:end_idx]
        
        # 确定数据块的标签
        label = 1 if np.any(label_slice == 1) else 0
        
        # 转换为torch张量
        chunk = torch.from_numpy(chunk_features)
        label = torch.tensor(label).long()

        if self.transform:
            chunk = self.transform(chunk)
            
        return chunk, label

if __name__ == '__main__':
    from torch.utils.data import random_split
    
    # 使用示例：
    dataset_root = './data/road'

    if not os.path.exists(dataset_root):
        print(f"数据集路径 '{dataset_root}' 未找到。")
    else:
        print("正在初始化ROADDataset...")
        # num_features = 1 ID + 22 signals
        road_dataset = ROADDataset(data_path=dataset_root, chunk_size=100, num_features=23)
        
        if len(road_dataset) > 0:
            dataset_size = len(road_dataset)
            print(f"\n数据集初始化成功。可生成的总数据块数: {dataset_size}")
            
            first_chunk, first_label = road_dataset[0]
            print(f"单个数据块的形状: {first_chunk.shape}")
            print(f"第一个数据块的标签: {first_label}")
            
            # 划分为训练、验证和测试集
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\n数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
            
            # 注意：random_split会创建一个索引列表，对于超大数据集可能会消耗一些内存
            # 但它比复制整个数据集要高效得多
            train_dataset, val_dataset, test_dataset = random_split(road_dataset, [train_size, val_size, test_size])
            
            # 创建DataLoader
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print("\nDataLoader创建成功！现在可以开始训练。")
            
            print("\n从train_loader中取一个批次进行测试...")
            for chunks_batch, labels_batch in train_loader:
                print(f"一个批次数据块的形状: {chunks_batch.shape}") # 应该是 [64, 100, 23]
                print(f"一个批次标签的形状: {labels_batch.shape}")   # 应该是 [64]
                break
        else:
            print("未能生成任何数据块。请检查数据路径和文件。") 