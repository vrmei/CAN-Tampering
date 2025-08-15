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
    Parse a single line of candump log.
    Log format: (timestamp) can_interface ID#DATA
    Example: (1602162391.566168) can0 110#02202E1300181300
    """
    match = re.match(r'\((\d+\.\d+)\)\s+\w+\s+([0-9A-F]+)#([0-9A-F]*)', line)
    if not match:
        return None, None, None
        
    timestamp = float(match.group(1))
    can_id_hex = match.group(2)
    can_id = int(can_id_hex, 16)
    
    data_hex = match.group(3).ljust(16, '0') # Pad to 8 bytes (16 hex characters)
    data = [int(data_hex[i:i+2], 16) for i in range(0, 16, 2)]
    
    return timestamp, can_id, data

def get_attack_intervals(json_path):
    """
    Parse attack start and end times from JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    markers = data.get('markers', [])
    starts = sorted([m['time'] for m in markers if 'Start' in m['description']])
    ends = sorted([m['time'] for m in markers if 'End' in m['description']])
    
    # Pair start and end times
    if len(starts) != len(ends) or not starts or not ends:
        print(f"Warning: Found mismatched start/end markers in {json_path}.")
        return []
        
    return list(zip(starts, ends))

def load_crysys_data(data_path, num_features=9):
    """
    Load all CrySyS log files and process features and labels.
    
    Args:
        data_path (str): Root directory path of CrySyS dataset.
        num_features (int): Number of features (1 ID + 8 Data Bytes).
        
    Returns:
        tuple: (all_features, all_labels)
    """
    print("Scanning log files...")
    all_log_files = glob.glob(os.path.join(data_path, '**/*.log'), recursive=True)
    
    all_features_list = []
    all_labels_list = []

    print("Processing log files...")
    for log_file in tqdm(all_log_files, desc="Files"):
        is_malicious = 'malicious' in os.path.basename(log_file).lower()
        attack_intervals = []
        
        if is_malicious:
            log_basename_full = os.path.basename(log_file)
            dir_name = os.path.dirname(log_file)
            
            # Handle special filename "-inj-messages.log"
            if log_basename_full.endswith('-inj-messages.log'):
                log_basename_stem = log_basename_full.replace('-inj-messages.log', '')
            else:
                log_basename_stem = log_basename_full.replace('.log', '')

            # Find matching JSON file
            json_file_path_to_find = os.path.join(dir_name, log_basename_stem + '.json')
            potential_json_files = glob.glob(json_file_path_to_find)
            
            if potential_json_files:
                # Usually only one best match
                json_file = potential_json_files[0]
                attack_intervals = get_attack_intervals(json_file)
            else:
                print(f"Warning: Cannot find JSON file corresponding to {log_file} (tried to find: {json_file_path_to_find})")
                continue

        with open(log_file, 'r') as f:
            for line in f:
                timestamp, can_id, data = parse_log_line(line.strip())
                if timestamp is None:
                    continue
                
                features = [can_id] + data
                # Ensure correct feature dimensions
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
    Memory-efficient CrySyS dataset PyTorch Dataset class.
    """
    def __init__(self, data_path, chunk_size=100, num_features=9, transform=None):
        """
        Args:
            data_path (str): Root directory path of CrySyS dataset.
            chunk_size (int): Size of each data chunk.
            num_features (int): Number of features per timestep (1 ID + 8 Data).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        features_cache_path = os.path.join(data_path, 'crysys_features.npy')
        labels_cache_path = os.path.join(data_path, 'crysys_labels.npy')

        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("Loading preprocessed CrySyS data from cache...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("Cache files not found, loading and preprocessing all CrySyS log data...")
            self.all_features, self.all_labels = load_crysys_data(data_path, num_features=num_features)
            
            print("Caching processed data as .npy files...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        
        if len(self.all_features) == 0:
            print("Error: Unable to load any data. Please check the path and file format.")
            return

        print(f"\nCrySyS data loading completed. Total timesteps: {len(self.all_features)}")
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
    from torch.utils.data import random_split
    
    dataset_root = './MIDS/data/CrySyS'

    if not os.path.exists(dataset_root):
        print(f"Dataset path '{dataset_root}' not found.")
    else:
        print("Initializing CrySySDataset...")
        # num_features = 1 ID + 8 data bytes
        crysys_dataset = CrySySDataset(data_path=dataset_root, chunk_size=100, num_features=9)
        
        if len(crysys_dataset) > 0:
            dataset_size = len(crysys_dataset)
            print(f"\nDataset initialization successful. Total data chunks available: {dataset_size}")
            
            first_chunk, first_label = crysys_dataset[0]
            print(f"Single data chunk shape: {first_chunk.shape}")
            print(f"First data chunk label: {first_label}")
            
            # Split into training, validation and test sets
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\nDataset size: {dataset_size}, Train: {train_size}, Validation: {val_size}, Test: {test_size}")
            
            train_dataset, val_dataset, test_dataset = random_split(crysys_dataset, [train_size, val_size, test_size])
            
            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print("\nDataLoader created successfully! Ready to start training.")
            
            print("\nTesting by taking one batch from train_loader...")
            # Use try-except block to handle num_workers issues on some systems
            try:
                for chunks_batch, labels_batch in train_loader:
                    print(f"One batch data chunks shape: {chunks_batch.shape}") # Should be [64, 100, 9]
                    print(f"One batch labels shape: {labels_batch.shape}")   # Should be [64]
                    break
            except Exception as e:
                print(f"Error occurred while testing DataLoader: {e}")
                print("Trying without multi-threaded workers (num_workers=0)...")
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
                for chunks_batch, labels_batch in train_loader:
                    print(f"One batch data chunks shape (no multithreading): {chunks_batch.shape}")
                    print(f"One batch labels shape (no multithreading): {labels_batch.shape}")
                    break

        else:
            print("Unable to generate any data chunks. Please check data path and files.") 