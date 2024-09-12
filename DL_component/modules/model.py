import torch
import torch.nn as nn
import numpy as np
from collections import Counter

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
            nn.Linear(in_features=128, out_features=1),
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
    
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        初始化SVM模型
        :param learning_rate: 学习率
        :param lambda_param: 正则化参数（惩罚项系数）
        :param n_iters: 迭代次数
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def fit(self, X, y):
        """
        训练SVM模型
        :param X: 训练样本的特征矩阵，shape (n_samples, n_features)
        :param y: 训练样本的标签向量，shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        # 将标签y转换为-1和1
        y_ = np.where(y <= 0, -1, 1)

        # 梯度下降法进行参数更新
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 计算合页损失，如果样本未正确分类
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # 如果分类正确，不更新权重，只更新正则化项
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # 如果分类错误，更新权重和偏置
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        对输入样本进行预测
        :param X: 测试样本的特征矩阵
        :return: 预测的标签
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)  # 返回预测类别：+1 或 -1

    def decision_function(self, X):
        """
        返回SVM的决策值，用于了解样本距离分类超平面的距离
        :param X: 测试样本的特征矩阵
        :return: 决策值
        """
        return np.dot(X, self.w) + self.b