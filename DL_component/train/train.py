import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn
import random
from tqdm import tqdm
from modules.model import Conv1D_Attention_Advanced, DeepConv1D_Attention, AttackDetectionModel
import logging
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
# Set random seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Setup logging
log_file = "training_log.txt"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", type=str, default='CE', help="Loss function type (MSE, CE)")
parser.add_argument("--model_type", type=str, default='CNN_Attention', help="The model to use (only CNN_Attention supported)")
parser.add_argument("--data_src", type=str, default='own', help="Dataset name")
parser.add_argument("--propotion", type=float, default=0.8, help="Proportion of training data")
parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--n_classes", type=int, default=4, help="Number of output classes")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--k_folds", type=int, default=5, help="Number of K-folds for cross-validation")
opt = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
path = 'data\owndata\merged\high-speed_merged.csv'
source_data = pd.read_csv(path)
data = source_data.iloc[:, :-1]
label = source_data.iloc[:, -1]

# Dataset class
class GetDataset(Dataset):  # 改为直接继承 torch.utils.data.Dataset
    def __init__(self, data_root, data_label):
        self.data = torch.tensor(data_root.values, dtype=torch.float32)
        self.label = torch.tensor(data_label.values, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# K-Fold Cross Validation
kf = KFold(n_splits=opt.k_folds, shuffle=True, random_state=42)
def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0
# Training and Evaluation Loop for K-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    logging.info(f"\nTraining fold {fold + 1}/{opt.k_folds}")

    # Split data into train and validation sets for this fold
    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    train_label, val_label = label.iloc[train_idx], label.iloc[val_idx]

    # Create DataLoader for this fold
    torch_data_train = GetDataset(train_data, train_label)
    torch_data_val = GetDataset(val_data, val_label)
    
    traindataloader = DataLoader(torch_data_train, batch_size=64, shuffle=True)
    valdataloader = DataLoader(torch_data_val, batch_size=64, shuffle=False)

    # Initialize CNN_Attention model for each fold
    model = DeepConv1D_Attention(input_width=900, num_classes=4).to(device)
    model = AttackDetectionModel(num_classes=4).to(device)

    # Loss function
    if opt.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif opt.loss_type == 'CE':
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.arange(opt.n_classes), 
            y=np.concatenate([labels.numpy() for _, labels in traindataloader])
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training Loop for the current fold
    for epoch in range(opt.n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        logging.info(f"Epoch [{epoch + 1}/{opt.n_epochs}]")
        train_bar = tqdm(traindataloader, desc="Training", leave=False)

        for data_x, data_y in train_bar:
            data_x, data_y = data_x.to(device), data_y.to(device)
            data_y = data_y.to(torch.long)

            optimizer.zero_grad()
            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * data_y.size(0)
            train_correct += (predicted == data_y).sum().item()
            train_total += data_y.size(0)

            train_bar.set_postfix(loss=train_loss / train_total, accuracy=train_correct / train_total)

        train_accuracy = train_correct / train_total
        logging.info(f"Training Loss: {train_loss / train_total:.4f}, Accuracy: {train_accuracy:.4f}")

        # Evaluation Loop for the current fold
        model.eval()
        val_correct = 0
        val_total = 0
        num_classes = opt.n_classes
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        val_bar = tqdm(valdataloader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for data_x, data_y in val_bar:
                data_x, data_y = data_x.to(device), data_y.to(device)
                outputs = model(data_x)
                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == data_y).sum().item()
                val_total += data_y.size(0)

                for cls in range(num_classes):
                    tp[cls] += ((predicted == cls) & (data_y == cls)).sum().item()
                    fp[cls] += ((predicted == cls) & (data_y != cls)).sum().item()
                    fn[cls] += ((predicted != cls) & (data_y == cls)).sum().item()

        # Calculate Precision, Recall, F1 for each class
        for cls in range(num_classes):
            precision = safe_division(tp[cls], tp[cls] + fp[cls])
            recall = safe_division(tp[cls], tp[cls] + fn[cls])
            f1 = safe_division(2 * precision * recall, precision + recall)
            logging.info(f"Class {cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

        # Compute Macro and Micro averages
        macro_precision = sum([safe_division(tp[cls], tp[cls] + fp[cls]) for cls in range(num_classes)]) / num_classes
        macro_recall = sum([safe_division(tp[cls], tp[cls] + fn[cls]) for cls in range(num_classes)]) / num_classes
        macro_f1 = sum([
            safe_division(
                2 * safe_division(tp[cls], tp[cls] + fp[cls]) * safe_division(tp[cls], tp[cls] + fn[cls]),
                safe_division(tp[cls], tp[cls] + fp[cls]) + safe_division(tp[cls], tp[cls] + fn[cls])
            )
            for cls in range(num_classes)
        ]) / num_classes

        micro_tp = sum(tp)
        micro_fp = sum(fp)
        micro_fn = sum(fn)
        micro_precision = safe_division(micro_tp, micro_tp + micro_fp)
        micro_recall = safe_division(micro_tp, micro_tp + micro_fn)
        micro_f1 = safe_division(2 * micro_precision * micro_recall, micro_precision + micro_recall)

        logging.info(f"Macro-Averaged Metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1 Score={macro_f1:.4f}")
        logging.info(f"Micro-Averaged Metrics: Precision={micro_precision:.4f}, Recall={micro_recall:.4f}, F1 Score={micro_f1:.4f}")

    # Save the model after each fold
    model_save_path = f"./output/{opt.model_type}_{opt.data_src}_fold{fold + 1}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at {model_save_path}")