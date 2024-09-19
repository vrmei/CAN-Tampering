import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ANN, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 定义输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        # 定义激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 前向传播：输入 -> 全连接层1 -> 激活函数 -> 全连接层2 -> 激活函数 -> 输出层
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
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


class TransformerClassifier_noposition(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes, dropout=0.1):
        """
        初始化Transformer分类模型，不使用词嵌入层，使用线性层直接处理输入特征
        :param input_dim: 输入特征的维度（如图像或时间序列的特征维度）
        :param num_heads: 多头自注意力机制中的头数
        :param num_layers: Transformer Encoder层的数量
        :param hidden_dim: Transformer中的隐藏层维度
        :param num_classes: 分类的类别数量
        :param dropout: Dropout比率
        """
        super(TransformerClassifier_noposition, self).__init__()

        # 输入线性投影：将输入的维度 input_dim 映射到 Transformer 的 hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类器
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据, 形状为 (batch_size, seq_len, input_dim)
        :return: 分类结果
        """
        # 投影到 Transformer 的 hidden_dim
        x = x.unsqueeze(1)
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)

        # Transformer 编码
        x = self.transformer_encoder(x)

        # 池化：可以用平均池化或只取第一个token（CLS token）的输出
        x = x.mean(dim=1)  # 平均池化

        # 全连接层进行分类
        output = self.fc(x)

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
            nn.Linear(in_features=128, out_features=2),
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