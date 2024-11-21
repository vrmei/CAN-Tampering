import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

path = 'data/owndata/processed/standby/standby1_processed.csv'
data = pd.read_csv(path, header=None)  # 假设没有列名，添加 header=None

# 转换为 NumPy 数组
data = data.values

# 每100行拼接为一行
rows_per_group = 100
reshaped_data = data[:rows_per_group * 1024].reshape(1024, -1)  # 保证数据可以整除，得到1024行

# 创建 X, Y 网格
x = np.arange(reshaped_data.shape[1])  # 列索引 (0-899)
y = np.arange(reshaped_data.shape[0])  # 行索引 (0-1023)
x, y = np.meshgrid(x, y)

# 数据值作为高度 Z
z = reshaped_data

# 创建 3D 图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label('Data Value (0-2047)')

# 设置轴标签
ax.set_xlabel('Columns (900)')
ax.set_ylabel('Rows (1024)')
ax.set_zlabel('Data Value')

# 设置标题
ax.set_title('Standby Data Matrix')

# 显示图像
plt.show()
