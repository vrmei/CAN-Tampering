import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
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
        super(TransformerClassifier, self).__init__()

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
