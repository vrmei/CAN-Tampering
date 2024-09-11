import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """
        初始化KNN模型
        :param k: 最近邻的数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        训练KNN模型，其实是保存训练数据，因为KNN是惰性学习
        :param X_train: 训练样本的特征矩阵
        :param y_train: 训练样本的标签向量
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        对测试样本进行预测
        :param X_test: 测试样本的特征矩阵
        :return: 预测的标签
        """
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        对单个样本进行预测
        :param x: 单个测试样本
        :return: 预测的标签
        """
        # 计算距离：欧氏距离
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 找到最近的K个邻居
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        """
        计算欧氏距离
        :param x1: 样本1
        :param x2: 样本2
        :return: 欧氏距离
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))