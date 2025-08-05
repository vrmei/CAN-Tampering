import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def load_can_train_and_test_data(data_path, num_features=3):
    """
    加载 can-train-and-test 数据集下所有csv文件，处理特征和标签。
    Args:
        data_path (str): can-train-and-test 数据集的根目录路径。
        num_features (int): 特征数量（1 arbitration_id + 2 data_field 拆分为2列，或可自定义）。
    Returns:
        tuple: (all_features, all_labels)
    """
    # 修正为递归查找所有csv文件
    all_csv_files = glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True)
    print(f'找到csv文件数量: {len(all_csv_files)}')
    all_features_list = []
    all_labels_list = []

    print("正在处理 can-train-and-test 所有csv文件...")
    for csv_file in tqdm(all_csv_files, desc="Files"):
        df = pd.read_csv(csv_file)
        
        # 处理 data_field 中的缺失值，用空字符串填充
        df['data_field'] = df['data_field'].fillna('')

        # arbitration_id 16进制转十进制
        df['arbitration_id'] = df['arbitration_id'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))
        # data_field 拆分为字节特征
        max_bytes = 8  # 最多8字节
        def split_data_field(s):
            s = str(s)
            s = s.ljust(max_bytes*2, '0')[:max_bytes*2]  # 补齐/截断
            return [int(s[i:i+2], 16) for i in range(0, max_bytes*2, 2)]
        data_bytes = df['data_field'].apply(split_data_field)
        data_bytes = np.stack(data_bytes.values)
        # 拼接 arbitration_id + data_bytes
        features = np.hstack([
            df['arbitration_id'].values.reshape(-1, 1),
            data_bytes
        ])
        labels = df['attack'].values.astype(np.uint8)
        all_features_list.append(features)
        all_labels_list.append(labels)
    all_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels_list, axis=0).astype(np.uint8)
    return all_features, all_labels


class CanTrainAndTestDataset(Dataset):
    """
    can-train-and-test 数据集的 PyTorch Dataset 类。
    """
    def __init__(self, data_path, chunk_size=100, num_features=9, transform=None):
        self.chunk_size = chunk_size
        self.transform = transform
        features_cache_path = os.path.join(data_path, 'can_train_and_test_features.npy')
        labels_cache_path = os.path.join(data_path, 'can_train_and_test_labels.npy')
        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("正在从缓存加载预处理的 can-train-and-test 数据...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("缓存文件未找到，正在加载和预处理所有 can-train-and-test csv 数据...")
            self.all_features, self.all_labels = load_can_train_and_test_data(data_path, num_features=num_features)
            print("正在缓存处理好的数据为.npy文件...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        if len(self.all_features) == 0:
            print("错误：未能加载任何数据。请检查路径和文件格式。")
            return
        print(f"\ncan-train-and-test 数据加载完成。总时间步: {len(self.all_features)}")
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
    from torch.utils.data import random_split, DataLoader
    dataset_root = './data/can-train-and-test'
    if not os.path.exists(dataset_root):
        print(f"数据集路径 '{dataset_root}' 未找到。")
    else:
        print("正在初始化 CanTrainAndTestDataset ...")
        dataset = CanTrainAndTestDataset(data_path=dataset_root, chunk_size=100, num_features=9)
        if len(dataset) > 0:
            dataset_size = len(dataset)
            print(f"\n数据集初始化成功。可生成的总数据块数: {dataset_size}")
            first_chunk, first_label = dataset[0]
            print(f"单个数据块的形状: {first_chunk.shape}")
            print(f"第一个数据块的标签: {first_label}")
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            print(f"\n数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
            print("\nDataLoader 创建成功！现在可以开始训练。")
