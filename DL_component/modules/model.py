import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from transformers import BertModel, BertConfig
import math

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.5):
        super(MLP, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 定义第一个层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 定义第二个层归一化
        self.ln2 = nn.LayerNorm(hidden_dim)
        # 定义输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义 Dropout，概率可以在实例化时传递，默认值为 0.5
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # 前向传播：
        # 输入 -> 全连接层1 -> 层归一化1 -> ReLU -> Dropout -> 全连接层2 -> 层归一化2 -> ReLU -> Dropout -> 输出层
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply Dropout after ReLU
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply Dropout after ReLU
        out = self.output_layer(out)
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 定义输出层
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x 的形状：(batch_size, seq_length, input_dim)
        # 初始化隐藏状态和细胞状态为零
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)
        return out

class PositionalEncodingWithDecay(nn.Module):
    def __init__(self, max_seq_len, embedding_dim, decay_interval=9, decay_factor=0.9):
        super(PositionalEncodingWithDecay, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        decay_steps = torch.floor(position / decay_interval).long()
        decay = decay_factor ** decay_steps.float()
        
        pe = pe * decay
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        pe = self.pe[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pe
        return x

class TransformerClassifier_WithPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes, 
                 embedding_dim=128, dropout=0.1, max_seq_len=600):
        """
        初始化Transformer分类模型，加入带有衰减机制的位置编码，使用Layer Normalization层并优化结构。
        
        :param input_dim: 输入特征的维度（如图像或时间序列的特征维度）
        :param num_heads: 多头自注意力机制中的头数
        :param num_layers: Transformer Encoder层的数量
        :param hidden_dim: Transformer中的隐藏层维度
        :param num_classes: 分类的类别数量
        :param embedding_dim: 嵌入维度
        :param dropout: Dropout比率
        :param max_seq_len: 序列的最大长度，用于位置编码
        """
        super(TransformerClassifier_WithPositionalEncoding, self).__init__()

        # 嵌入层：将 input_dim 映射到 embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)
        # 位置编码
        self.positional_encoding = PositionalEncodingWithDecay(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            decay_interval=9,
            decay_factor=0.9
        )

        # LayerNorm层
        self.layer_norm_input = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu'  # 使用ReLU激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据, 形状为 (batch_size, seq_len, input_dim)
        :return: 分类结果
        """
        batch_size, seq_len, input_dim = x.size()

        # 嵌入层
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim=128)

        # 添加位置编码
        x = self.positional_encoding(x)  # (batch_size, seq_len, embedding_dim=128)

        # LayerNorm
        x = self.layer_norm_input(x)  # (batch_size, seq_len, embedding_dim=128)
        x = torch.relu(x)              # 激活函数
        x = self.dropout(x)

        # Transformer 编码
        # Transformer期望的输入形状为 (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

        # 池化：平均池化
        x = x.mean(dim=1)  # (batch_size, embedding_dim)

        # 分类器进行分类
        output = self.fc(x)  # (batch_size, num_classes)

        return output




class BERT(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes, 
                 embedding_dim=128, dropout=0.1, max_seq_len=600):
        """
        初始化Transformer分类模型，加入带有衰减机制的位置编码，使用Layer Normalization层并优化结构。
        
        :param input_dim: 输入特征的维度（如图像或时间序列的特征维度）
        :param num_heads: 多头自注意力机制中的头数
        :param num_layers: Transformer Encoder层的数量
        :param hidden_dim: Transformer中的隐藏层维度
        :param num_classes: 分类的类别数量
        :param embedding_dim: 嵌入维度
        :param dropout: Dropout比率
        :param max_seq_len: 序列的最大长度，用于位置编码
        """
        super(BERT, self).__init__()

        # 嵌入层：将 input_dim 映射到 embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # 位置编码
        self.positional_encoding = PositionalEncodingWithDecay(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            decay_interval=9,
            decay_factor=0.9
        )

        # LayerNorm层
        self.layer_norm_input = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # 定义8个相同的Transformer Encoder层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu'  # 使用ReLU激活函数
            )
            for _ in range(8)  # 连接8个相同的Encoder层
        ])

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据, 形状为 (batch_size, seq_len, input_dim)
        :return: 分类结果
        """
        batch_size, seq_len, input_dim = x.size()

        # 嵌入层
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim=128)

        # 添加位置编码
        x = self.positional_encoding(x)  # (batch_size, seq_len, embedding_dim=128)

        # LayerNorm
        x = self.layer_norm_input(x)  # (batch_size, seq_len, embedding_dim=128)
        x = torch.relu(x)              # 激活函数
        x = self.dropout(x)

        # Transformer 编码
        # Transformer期望的输入形状为 (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        for encoder in self.encoder_layers:
            x = encoder(x)  # (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

        # 池化：平均池化
        x = x.mean(dim=1)  # (batch_size, embedding_dim)

        # 分类器进行分类
        output = self.fc(x)  # (batch_size, num_classes)

        return output




class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=288, out_features=2),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
    
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConv1D_Attention(nn.Module):
    def __init__(self, input_width, num_classes, embedding_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.2):
        """
        Model combining deep Conv1D (short and long distance) and Attention for feature extraction and classification.
        
        Args:
        - input_width (int): Width of the input sequence (e.g., 900).
        - num_classes (int): Number of output classes.
        - embedding_dim (int): Dimension of embeddings for attention mechanism.
        - num_heads (int): Number of attention heads.
        - hidden_dim (int): Dimension of the feedforward layer in attention.
        - num_layers (int): Number of Transformer layers.
        - dropout (float): Dropout rate.
        """
        super(DeepConv1D_Attention, self).__init__()
        
        # Short-distance Conv1D with multiple layers
        self.short_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=0),  # Short filter
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduce width by half
        )
        
        # Long-distance Conv1D with multiple layers
        self.long_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=90, stride=1, padding=0),  # Long filter
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduce width by half
        )
        
        # Compute feature dimensions dynamically
        short_out_dim = (input_width // 9 // 2) * 64  # Output size from short Conv1D
        long_out_dim = ((input_width // 9 - 9 + 1) // 2) * 64  # Output size from long Conv1D
        original_out_dim = input_width  # Flattened original input
        
        feature_dim = 900 #55364  # Total concatenated features
        
        # Linear layer to map concatenated features to embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, width=900).
        
        Returns:
        - torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, width = x.shape  # Shape: (batch_size, 900)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 900)
        
        # Short-distance Conv1D
        x_short = self.short_conv1d(x)  # Shape: (batch_size, 64, width/18)
        x_short = x_short.view(batch_size, -1)  # Flatten: (batch_size, 64 * width/18)
        
        # Long-distance Conv1D
        x_long = self.long_conv1d(x)  # Shape: (batch_size, 64, (width/9 - 9 + 1)/2)
        x_long = x_long.view(batch_size, -1)  # Flatten: (batch_size, 64 * (width/9 - 9 + 1)/2)
        
        # Flatten input for concatenation
        x_flat = x.view(batch_size, -1)  # Shape: (batch_size, 900)
        
        # Concatenate original, short, and long features
        x_concat = torch.cat([x_flat, x_short, x_long], dim=1)  # Shape: (batch_size, feature_dim)
        
        # Linear projection to embedding_dim
        x = self.embedding_layer(x_flat)  # Shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, seq_len=1, embedding_dim)
        
        # Attention
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)
        x = self.attention(x)  # Shape: (seq_len=1, batch_size, embedding_dim)
        x = x.permute(1, 0, 2).squeeze(1)  # Back to (batch_size, embedding_dim)
        
        # Classification
        out = self.classifier(x)  # Shape: (batch_size, num_classes)
        out = torch.softmax(out, dim=1)  # Convert logits to probabilities
        return out


class Conv1D_Attention_Advanced(nn.Module):
    def __init__(self, input_width, num_classes, embedding_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.2):
        """
        Enhanced model combining Conv1D (short and long distance) and Attention for feature extraction and classification.
        
        Args:
        - input_width (int): Width of the input sequence (e.g., 900).
        - num_classes (int): Number of output classes.
        - embedding_dim (int): Dimension of embeddings for attention mechanism.
        - num_heads (int): Number of attention heads.
        - hidden_dim (int): Dimension of the feedforward layer in attention.
        - num_layers (int): Number of Transformer layers.
        - dropout (float): Dropout rate.
        """
        super(Conv1D_Attention_Advanced, self).__init__()
        
        # Short-distance 1D Convolutional Layers
        self.short_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduce width by half
            nn.Dropout(dropout)
        )
        
        # Long-distance 1D Convolutional Layers
        self.long_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduce width by half
            nn.Dropout(dropout)
        )
        
        # Linear layer to map concatenated features to embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(64 * (input_width // 2) + 64 * (input_width // 2), embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with additional hidden layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, width=900).
        
        Returns:
        - torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, width = x.shape  # Shape: (batch_size, 900)
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1D (batch_size, 1, 900)
        
        # Short-distance 1D Convolution
        x_short = self.short_conv1d(x)  # Shape: (batch_size, 64, width/2)
        
        # Long-distance 1D Convolution
        x_long = self.long_conv1d(x)  # Shape: (batch_size, 64, width/2)
        
        # Flatten Conv1D outputs
        x_short = x_short.view(batch_size, -1)  # Shape: (batch_size, 64 * width/2)
        x_long = x_long.view(batch_size, -1)  # Shape: (batch_size, 64 * width/2)
        
        # Concatenate short and long 1D Conv features
        x_concat = torch.cat([x_short, x_long], dim=1)  # Shape: (batch_size, 64 * width/2 + 64 * width/2)
        
        # Linear projection to embedding_dim
        x = self.embedding_layer(x_concat)  # Shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, embedding_dim)
        
        # Attention
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)
        x = self.attention(x)  # Shape: (seq_len=1, batch_size, embedding_dim)
        x = x.permute(1, 0, 2).squeeze(1)  # Back to (batch_size, embedding_dim)
        
        # Classification
        out = self.classifier(x)  # Shape: (batch_size, num_classes)
        return out

# 旋转位置编码（ROPE）实现
def rope_position_encoding(seq_len, dim):
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim//2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
    return pe.unsqueeze(0)  # Shape: (1, seq_len, dim)

# 注意力机制实现
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x  # Residual connection
        x = self.layer_norm(x)

        # Feed-forward network
        x = self.fc(x) + attn_output  # Residual connection
        return x

# 修改后的AttackDetectionModel
class AttackDetectionModel(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2, num_heads=4):
        super(AttackDetectionModel, self).__init__()

        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple 1D convolutions for data
        self.data_conv1 = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.data_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)

        # Attention Layer
        self.attn_layer = AttentionLayer(input_dim=embed_dim + hidden_dim * 2, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Position encoding (ROPE)
        self.pos_encoding = rope_position_encoding(100, embed_dim + hidden_dim * 2)  # Assume sequence length is 100
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv1(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = F.relu(self.data_conv2(data_feat))  # Shape: (batch_size, hidden_dim*2, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim*2)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Add positional encoding (ROPE)
        combined_feat = combined_feat + self.pos_encoding  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Apply Attention
        attention_out = self.attn_layer(combined_feat)  # Shape: (batch_size, 100, embed_dim + hidden_dim*2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(attention_out, dim=1).values  # Shape: (batch_size, embed_dim + hidden_dim*2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out
    

# class AttackDetectionModel(nn.Module):
    def __init__(self, input_width=1, embed_dim=256, data_dim=8, hidden_dim=64, num_classes=4, dropout=0.2):
        super(AttackDetectionModel, self).__init__()
        # Embedding layer for IDs
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 1D convolution for data
        self.data_conv = nn.Conv1d(in_channels=data_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=embed_dim + hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input (batch_size, 900) -> (batch_size, 100, 9)
        x = x.view(x.size(0), 100, 9)
        
        # Split ID and data
        id_data = x[:, :, 0]  # Shape: (batch_size, 100)
        data = x[:, :, 1:]    # Shape: (batch_size, 100, 8)
        
        # Add an extra dimension to id_data for embedding
        id_data = id_data.unsqueeze(-1)  # Shape: (batch_size, 100, 1)
        
        # Process ID with embedding
        id_embed = self.embedding_layer(id_data)  # Shape: (batch_size, 100, embed_dim)
        
        # Process data with convolution
        data = data.permute(0, 2, 1)  # Change to (batch_size, 8, 100)
        data_feat = F.relu(self.data_conv(data))  # Shape: (batch_size, hidden_dim, 100)
        data_feat = data_feat.permute(0, 2, 1)  # Change back to (batch_size, 100, hidden_dim)
        
        # Concatenate ID and data features
        combined_feat = torch.cat((id_embed, data_feat), dim=-1)  # Shape: (batch_size, 100, embed_dim + hidden_dim)
        
        # Sequence modeling with LSTM
        lstm_out, _ = self.lstm(combined_feat)  # Shape: (batch_size, 100, hidden_dim * 2)
        
        # Max pooling over sequence
        pooled_feat = torch.max(lstm_out, dim=1).values  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc(pooled_feat)  # Shape: (batch_size, num_classes)
        
        return out
