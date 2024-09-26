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