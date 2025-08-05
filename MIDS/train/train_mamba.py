import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn
import random
from tqdm import tqdm
from modules.model import AttackDetectionModel_Sincos, AttackDetectionModel_no_pos, AttackDetectionModel_no_attn, AttackDetectionModel_no_conv_pos, AttackDetectionModel_no_embedding_pos
from modules.model import AttackDetectionModel, AttackDetectionModel_Sincos_Fre_EMB, AttackDetectionModel_Sincos_Fre, AttackDetectionModel_LSTM
from modules.model import MambaCAN, MambaCAN_noconv, MambaCAN_noid, MambaCAN_Only, MambaCAN_2Direction, MambaCAN_2Direction_1conv, MambaCAN_2Direction_Fre
from data_loader import ROADDataset
from crysys_data_loader import CrySySDataset
from otids_data_loader import OTIDSDataset
from can_train_and_test_data_loader import CanTrainAndTestDataset
import logging
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch.nn.functional as F

# Enable anomaly detection to get more detailed error messages on NaN issues.
torch.autograd.set_detect_anomaly(True)

class FocalLoss(nn.Module):
    """
    The final, definitive, and logically correct implementation of Focal Loss.
    This version strictly follows the formula's logic by separating the calculation
    of the raw prediction probability (pt) from the application of the alpha and
    gamma terms. This ensures the computational logic is sound and robust.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha (Tensor): a manual rescaling weight given to each class.
        #               If given, has to be a Tensor of size C.
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Compute the unweighted cross-entropy loss first.
        # This is only to get the correct `pt` (prediction probability of the true class).
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Calculate pt from the unweighted ce_loss.
        pt = torch.exp(-ce_loss)
        
        # 3. Gather the alpha weights for each sample in the batch.
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            # Apply the alpha weight to the standard focal loss formula.
            focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        else:
            # If no alpha is provided, use the original focal loss formula.
            focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# Function to compute and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, fold, output_dir="./output"):
    """
    Plots and saves confusion matrix.
    Args:
        y_true (list or numpy array): Ground truth labels.
        y_pred (list or numpy array): Predicted labels.
        class_names (list): List of class names.
        fold (int): Fold index for file naming.
        output_dir (str): Directory to save the confusion matrix image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix for Fold {fold + 1}")
    plt.savefig(f"{output_dir}/confusion_matrix_fold_{fold + 1}_{log_file}.png")
    plt.close()

def plot_roc_curves(y_true, y_scores, n_classes, fold, output_dir="./output"):
    """
    Plots and saves ROC curves for each class and a macro-average. This provides
    a detailed view of the trade-off between detecting true positives and avoiding
    false alarms, which is critical for security applications.
    """
    if n_classes == 2:
        # For binary classification, compute ROC for the positive class (class 1)
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = {0:0.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill (Luck)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'Receiver Operating Characteristic (ROC) for Fold {fold + 1}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{output_dir}/roc_curve_fold_{fold + 1}_{log_file}.png")
        plt.close()
        return

    # Binarize the labels for multi-class ROC analysis
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    class_names=[f"Class {i}" for i in range(n_classes)]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill (Luck)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'Receiver Operating Characteristic (ROC) for Fold {fold + 1}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{output_dir}/roc_curve_fold_{fold + 1}_{log_file}.png")
    plt.close()


# Set random seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Setup logging
log_file = "ablation_AttackDetectionModel_no_embedding_pos_acc.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", type=str, default='CE', choices=['CE', 'Focal'], help="Loss function type (CE, Focal)")
parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--n_classes", type=int, default=2, help="Number of output classes")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for Focal Loss")
parser.add_argument("--k_folds", type=int, default=5, help="Number of K-folds for cross-validation")
parser.add_argument("--dataset", type=str, default='cantrainandtest', choices=['road', 'own', 'crysys', 'otids', 'cantrainandtest'], help="Dataset to use")
opt = parser.parse_args()

# Set device
device = torch.device("cuda:0")

# Dataset class
class GetDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        # Convert to tensor on the fly from the numpy/memmap array
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

if opt.dataset == 'road':
    # Load dataset
    dataset_root = os.path.join(parent_dir, 'data', 'road')
    full_dataset = ROADDataset(data_path=dataset_root, chunk_size=100, num_features=23)
elif opt.dataset == 'own':
    # Load dataset
    path = 'final_dataset/final_dataset.npy'
    source_data = np.load(path)
    data = source_data[:, :-1]
    label = source_data[:, -1]
    full_dataset = GetDataset(data, label)
elif opt.dataset == 'crysys':
    dataset_root = os.path.join(parent_dir, 'data', 'CrySyS')
    # For CrySyS, num_features = 1 ID + 8 Data bytes = 9
    full_dataset = CrySySDataset(data_path=dataset_root, chunk_size=100, num_features=9)
elif opt.dataset == 'otids':
    dataset_root = os.path.join(parent_dir, 'data', 'OTIDS')
    full_dataset = OTIDSDataset(data_path=dataset_root, chunk_size=100)
elif opt.dataset == 'cantrainandtest':
    dataset_root = os.path.join(parent_dir, 'data', 'can-train-and-test')
    full_dataset = CanTrainAndTestDataset(data_path=dataset_root, chunk_size=100, num_features=9)


# K-Fold Cross Validation
kf = KFold(n_splits=opt.k_folds, shuffle=True, random_state=42)

def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

# Training and Evaluation Loop for K-Fold
sum_rec, sum_pre, sum_F1, sum_acc = 0, 0, 0, 0
sum_fpr, sum_fnr = 0, 0 # Initialize sums for new metrics
all_time = []

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
    logging.info(f"\nTraining fold {fold + 1}/{opt.k_folds}")

    # Create data subsets for this fold
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Create DataLoader for this fold from the loaded/created numpy arrays
    traindataloader = DataLoader(train_subset, batch_size=1024, shuffle=True)
    valdataloader = DataLoader(val_subset, batch_size=1024, shuffle=False)

    # Initialize the model for each fold
    if opt.dataset == 'road':
        model = MambaCAN_2Direction(num_classes=opt.n_classes, data_dim=22).to(device)
    elif opt.dataset == 'own':
        model = MambaCAN_2Direction(num_classes=opt.n_classes, data_dim=8).to(device)
    elif opt.dataset == 'crysys':
        model = MambaCAN_2Direction(num_classes=opt.n_classes, data_dim=8).to(device)
    elif opt.dataset == 'otids':
        model = MambaCAN_2Direction(num_classes=opt.n_classes, data_dim=8).to(device)
    elif opt.dataset == 'cantrainandtest':
        model = MambaCAN_2Direction(num_classes=opt.n_classes, data_dim=8).to(device)

    # Loss function
    if opt.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif opt.loss_type == 'CE':
        # Efficiently calculate class weights from the numpy array of labels
        if opt.dataset == 'own':
            y_train = full_dataset.label[train_idx]
        else:
            y_train = np.array([label for _, label in train_subset])
        
        possible_classes = np.arange(opt.n_classes)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=possible_classes,
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif opt.loss_type == 'Focal':
        # Calculate weights similarly for Focal Loss
        if opt.dataset == 'own':
            y_train = full_dataset.label[train_idx]
        else:
            y_train = np.array([label for _, label in train_subset])
        
        possible_classes = np.arange(opt.n_classes)

        class_weights = compute_class_weight(class_weight='balanced', classes=possible_classes, y=y_train)
        # The alpha parameter in FocalLoss can be a list of weights per class.
        # Note: A tensor of weights is passed to alpha.
        criterion = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=opt.gamma)

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
        total_time = 0.0
        num_items = len(train_bar)
        for data_x, data_y in train_bar:
            start_time = time.time()
            data_x, data_y = data_x.to(device), data_y.to(device)
            data_y = data_y.to(torch.long)

            optimizer.zero_grad()
            outputs = model(data_x)
            end_time = time.time()
            total_time += end_time - start_time

            loss = criterion(outputs, data_y)
            
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * data_y.size(0)
            train_correct += (predicted == data_y).sum().item()
            train_total += data_y.size(0)

            train_bar.set_postfix(loss=train_loss / train_total, accuracy=train_correct / train_total)

        train_accuracy = train_correct / train_total
        average_time = total_time / num_items
        logging.info(f"Training Loss: {train_loss / train_total:.4f}, Accuracy: {train_accuracy:.4f}")
        logging.info(f"Item time: {average_time:.4f}")
        all_time.append(average_time)

        # Evaluation Loop for the current fold
        model.eval()
        val_correct = 0
        val_total = 0
        num_classes = opt.n_classes
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        tn = [0] * num_classes # Added True Negatives

        all_labels = []
        all_preds = []
        all_scores = [] # Added for ROC curve analysis

        val_bar = tqdm(valdataloader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for data_x, data_y in val_bar:
                data_x, data_y = data_x.to(device), data_y.to(device)
                outputs = model(data_x)
                scores = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == data_y).sum().item()
                val_total += data_y.size(0)

                all_labels.extend(data_y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

                for cls in range(num_classes):
                    tp[cls] += ((predicted == cls) & (data_y == cls)).sum().item()
                    fp[cls] += ((predicted == cls) & (data_y != cls)).sum().item()
                    fn[cls] += ((predicted != cls) & (data_y == cls)).sum().item()
                    tn[cls] += ((predicted != cls) & (data_y != cls)).sum().item()

        val_accuracy = val_correct / val_total
        macro_precision = sum([safe_division(tp[cls], tp[cls] + fp[cls]) for cls in range(num_classes)]) / num_classes
        macro_recall = sum([safe_division(tp[cls], tp[cls] + fn[cls]) for cls in range(num_classes)]) / num_classes
        macro_fpr = sum([safe_division(fp[cls], fp[cls] + tn[cls]) for cls in range(num_classes)]) / num_classes
        macro_fnr = sum([safe_division(fn[cls], fn[cls] + tp[cls]) for cls in range(num_classes)]) / num_classes
        macro_f1 = sum([
            safe_division(
                2 * (macro_precision * macro_recall),
                (macro_precision + macro_recall)
            )
        ])

        if epoch == opt.n_epochs - 1:  # Last epoch
            sum_pre += macro_precision
            sum_rec += macro_recall
            sum_F1 += macro_f1
            sum_acc += val_accuracy
            sum_fpr += macro_fpr
            sum_fnr += macro_fnr

        logging.info(f"Macro-Averaged Metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1 Score={macro_f1:.4f}")
        logging.info(f"                       Accuracy={val_accuracy:.4f}, FPR={macro_fpr:.4f}, FNR={macro_fnr:.4f}")

    # Generate confusion matrix and ROC curve plots after all validation batches
    plot_confusion_matrix(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        class_names=[f"Class {i}" for i in range(num_classes)],
        fold=fold,
        output_dir="./output"
    )
    plot_roc_curves(
        y_true=np.array(all_labels),
        y_scores=np.array(all_scores),
        n_classes=num_classes,
        fold=fold,
        output_dir="./output"
    )
    # Save the model after each fold
    model_save_path = f"./output/{log_file}_fold{fold + 1}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at {model_save_path}")

# Final metrics
logging.info(f"Final Macro-Averaged Metrics across {opt.k_folds} Folds:")
logging.info(f"  Precision: {sum_pre / opt.k_folds:.4f}")
logging.info(f"  Recall:    {sum_rec / opt.k_folds:.4f}")
logging.info(f"  F1 Score:  {sum_F1 / opt.k_folds:.4f}")
logging.info(f"  Accuracy:  {sum_acc / opt.k_folds:.4f}")
logging.info(f"  FPR:       {sum_fpr / opt.k_folds:.4f}")
logging.info(f"  FNR:       {sum_fnr / opt.k_folds:.4f}")
logging.info("A lower FPR (fewer false alarms) and FNR (fewer missed attacks) are critical for a production-ready IDS.")
all_times_array = np.array(all_time)
logging.info(f"Average item time: {all_times_array.mean():.4f}")