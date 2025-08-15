import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def load_can_train_and_test_data(data_path, num_features=3):
    """
    Load all CSV files from the can-train-and-test dataset and process features and labels.
    Args:
        data_path (str): Root directory path of the can-train-and-test dataset.
        num_features (int): Number of features (1 arbitration_id + data_field split into columns, or customizable).
    Returns:
        tuple: (all_features, all_labels)
    """
    # Find all CSV files recursively
    all_csv_files = glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True)
    print(f'Found CSV files: {len(all_csv_files)}')
    all_features_list = []
    all_labels_list = []

    print("Processing all can-train-and-test CSV files...")
    for csv_file in tqdm(all_csv_files, desc="Files"):
        df = pd.read_csv(csv_file)
        
        # Handle missing values in data_field, fill with empty string
        df['data_field'] = df['data_field'].fillna('')

        # Convert arbitration_id from hexadecimal to decimal
        df['arbitration_id'] = df['arbitration_id'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))
        # Split data_field into byte features
        max_bytes = 8  # Maximum 8 bytes
        def split_data_field(s):
            s = str(s)
            s = s.ljust(max_bytes*2, '0')[:max_bytes*2]  # Pad/truncate
            return [int(s[i:i+2], 16) for i in range(0, max_bytes*2, 2)]
        data_bytes = df['data_field'].apply(split_data_field)
        data_bytes = np.stack(data_bytes.values)
        # Concatenate arbitration_id + data_bytes
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
    PyTorch Dataset class for can-train-and-test dataset.
    """
    def __init__(self, data_path, chunk_size=100, num_features=9, transform=None):
        self.chunk_size = chunk_size
        self.transform = transform
        features_cache_path = os.path.join(data_path, 'can_train_and_test_features.npy')
        labels_cache_path = os.path.join(data_path, 'can_train_and_test_labels.npy')
        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("Loading preprocessed can-train-and-test data from cache...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("Cache files not found, loading and preprocessing all can-train-and-test CSV data...")
            self.all_features, self.all_labels = load_can_train_and_test_data(data_path, num_features=num_features)
            print("Caching processed data as .npy files...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        if len(self.all_features) == 0:
            print("Error: Unable to load any data. Please check the path and file format.")
            return
        print(f"\ncan-train-and-test data loading completed. Total timesteps: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        print(f"Total attack timesteps: {attack_count}, Total normal timesteps: {normal_count}")
        print(f"Imbalance ratio (normal/attack): {imbalance_ratio:.2f}")

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
        print(f"Dataset path '{dataset_root}' not found.")
    else:
        print("Initializing CanTrainAndTestDataset...")
        dataset = CanTrainAndTestDataset(data_path=dataset_root, chunk_size=100, num_features=9)
        if len(dataset) > 0:
            dataset_size = len(dataset)
            print(f"\nDataset initialization successful. Total data chunks available: {dataset_size}")
            first_chunk, first_label = dataset[0]
            print(f"Single data chunk shape: {first_chunk.shape}")
            print(f"First data chunk label: {first_label}")
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            print(f"\nDataset size: {dataset_size}, Train: {train_size}, Validation: {val_size}, Test: {test_size}")
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
            print("\nDataLoader created successfully! Ready to start training.")
