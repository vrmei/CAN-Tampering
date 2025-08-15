import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def parse_otids_line(line):
    """
    Parse a single line from OTIDS dataset file.
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
    
    # Pad data to 8 bytes
    padded_data = data + [0] * (8 - len(data))
    
    # Features will be [ID, D0, D1, ..., D7]
    return [can_id] + padded_data

def load_all_otids_data(data_path):
    """
    Load all OTIDS .txt files, process features and labels,
    and return two Numpy arrays.
    """
    files = {
        "Attack_free_dataset.txt": 0,
        "Impersonation_attack_dataset.txt": 1,
        "Fuzzy_attack_dataset.txt": 1,
        "DoS_attack_dataset.txt": 1
    }
    
    all_features_list = []
    all_labels_list = []

    print("Processing OTIDS .txt files...")
    for filename, label in files.items():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
        
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        file_features = []
        for line in tqdm(lines, desc=f"Processing {filename}"):
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
    PyTorch Dataset class for OTIDS dataset.
    """
    def __init__(self, data_path, chunk_size=100, transform=None):
        """
        Args:
            data_path (str): Path to OTIDS dataset directory.
            chunk_size (int): Size of each data chunk (e.g., 100 timesteps).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        features_cache_path = os.path.join(data_path, 'otids_features.npy')
        labels_cache_path = os.path.join(data_path, 'otids_labels.npy')

        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("Loading preprocessed data from cache...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("Cache not found, loading and preprocessing all OTIDS data...")
            self.all_features, self.all_labels = load_all_otids_data(data_path)
            
            if self.all_features.size > 0:
                print("Caching processed data to .npy files...")
                np.save(features_cache_path, self.all_features)
                np.save(labels_cache_path, self.all_labels)
        
        if len(self.all_features) == 0:
             print("Warning: No data loaded.")
             return

        print(f"\nData loading completed. Total timesteps: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"Total attack timesteps: {attack_count}")
        print(f"Total normal timesteps: {normal_count}")
        print(f"Imbalance ratio (normal/attack): {imbalance_ratio:.2f}")

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
        print(f"Dataset path '{dataset_root}' not found.")
    else:
        print("Initializing OTIDSDataset...")
        otids_dataset = OTIDSDataset(data_path=dataset_root, chunk_size=100)
        
        if len(otids_dataset) > 0:
            dataset_size = len(otids_dataset)
            print(f"\nDataset initialization successful. Total data chunks: {dataset_size}")
            
            first_chunk, first_label = otids_dataset[0]
            print(f"Single data chunk shape: {first_chunk.shape}")
            print(f"First data chunk label: {first_label}")
            
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\nDataset size: {dataset_size}, Train: {train_size}, Validation: {val_size}, Test: {test_size}")
            
            train_dataset, val_dataset, test_dataset = random_split(otids_dataset, [train_size, val_size, test_size])
            
            # When using multiprocessing (num_workers > 0), it's recommended to set 'fork' start method on Linux
            # import torch.multiprocessing as mp
            # mp.set_start_method('fork', force=True)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True) # num_workers > 0 may cause issues in some environments
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

            print("\nDataLoader created successfully! Ready to start training.")
            
            print("\nTesting by taking one batch from train_loader...")
            try:
                for i, (chunks_batch, labels_batch) in enumerate(train_loader):
                    print(f"Batch {i+1} data chunks shape: {chunks_batch.shape}")
                    print(f"Batch {i+1} labels shape: {labels_batch.shape}")
                    if i >= 0: # Only test one batch
                        break
            except Exception as e:
                print(f"Error occurred during DataLoader iteration: {e}")
                print("Please check your environment. If num_workers > 0 causes issues, try setting it to 0.") 