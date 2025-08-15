import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_all_data(data_path, num_signals_needed=22):
    """
    Load all CSV files, process features and labels, and return two Numpy arrays.
    
    Args:
        data_path (str): Root directory path of ROAD dataset.
        num_signals_needed (int): Number of signal features needed.

    Returns:
        tuple: (all_features, all_labels)
    """
    ambient_files = glob.glob(os.path.join(data_path, 'signal_extractions/ambient/*.csv'))
    attack_files = glob.glob(os.path.join(data_path, 'signal_extractions/attacks/*.csv'))
    
    all_files = ambient_files + attack_files
    
    all_features_list = []
    all_labels_list = []

    print("Processing CSV files...")
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
    Memory-efficient ROAD dataset PyTorch Dataset class.
    Dynamically creates non-overlapping data chunks in __getitem__.
    """
    def __init__(self, data_path, chunk_size=100, num_features=23, transform=None):
        """
        Args:
            data_path (str): Root directory path of ROAD dataset.
            chunk_size (int): Size of each data chunk (e.g., 100 timesteps).
            num_features (int): Number of features per timestep (1 ID + 22 Signals).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.chunk_size = chunk_size
        self.transform = transform
        
        # Define cache file paths
        features_cache_path = os.path.join(data_path, 'all_features.npy')
        labels_cache_path = os.path.join(data_path, 'all_labels.npy')

        # Check if cache files exist
        if os.path.exists(features_cache_path) and os.path.exists(labels_cache_path):
            print("Loading preprocessed data from cache...")
            self.all_features = np.load(features_cache_path)
            self.all_labels = np.load(labels_cache_path)
        else:
            print("Cache files not found, loading and preprocessing all CSV data...")
            self.all_features, self.all_labels = load_all_data(data_path, num_signals_needed=num_features - 1)
            
            # Cache processed data as .npy files
            print("Caching processed data as .npy files...")
            np.save(features_cache_path, self.all_features)
            np.save(labels_cache_path, self.all_labels)
        
        print(f"\nData loading completed. Total timesteps: {len(self.all_features)}")
        attack_count = np.sum(self.all_labels)
        normal_count = len(self.all_labels) - attack_count
        
        # Calculate sample imbalance ratio
        imbalance_ratio = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"Total attack timesteps: {attack_count}, Total normal timesteps: {normal_count}")
        print(f"Imbalance ratio (normal/attack): {imbalance_ratio:.2f}")

    def __len__(self):
        # Total number of samples is total length divided by chunk size
        return len(self.all_features) // self.chunk_size

    def __getitem__(self, idx):
        # Calculate start and end indices for non-overlapping chunks
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # Dynamically slice data chunks from complete data
        chunk_features = self.all_features[start_idx:end_idx]
        label_slice = self.all_labels[start_idx:end_idx]
        
        # Determine the label for the data chunk
        label = 1 if np.any(label_slice == 1) else 0
        
        # Convert to torch tensors
        chunk = torch.from_numpy(chunk_features)
        label = torch.tensor(label).long()

        if self.transform:
            chunk = self.transform(chunk)
            
        return chunk, label

if __name__ == '__main__':
    from torch.utils.data import random_split
    
    # Usage example:
    dataset_root = './data/road'

    if not os.path.exists(dataset_root):
        print(f"Dataset path '{dataset_root}' not found.")
    else:
        print("Initializing ROADDataset...")
        # num_features = 1 ID + 22 signals
        road_dataset = ROADDataset(data_path=dataset_root, chunk_size=100, num_features=23)
        
        if len(road_dataset) > 0:
            dataset_size = len(road_dataset)
            print(f"\nDataset initialization successful. Total data chunks available: {dataset_size}")
            
            first_chunk, first_label = road_dataset[0]
            print(f"Single data chunk shape: {first_chunk.shape}")
            print(f"First data chunk label: {first_label}")
            
            # Split into training, validation and test sets
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            print(f"\nDataset size: {dataset_size}, Train: {train_size}, Validation: {val_size}, Test: {test_size}")
            
            # Note: random_split creates an index list, which may consume some memory for very large datasets
            # But it's much more efficient than copying the entire dataset
            train_dataset, val_dataset, test_dataset = random_split(road_dataset, [train_size, val_size, test_size])
            
            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print("\nDataLoader created successfully! Ready to start training.")
            
            print("\nTesting by taking one batch from train_loader...")
            for chunks_batch, labels_batch in train_loader:
                print(f"One batch data chunks shape: {chunks_batch.shape}") # Should be [64, 100, 23]
                print(f"One batch labels shape: {labels_batch.shape}")   # Should be [64]
                break
        else:
            print("Unable to generate any data chunks. Please check data path and files.") 