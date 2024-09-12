import numpy as np
import matplotlib.pyplot as plt

# 定义阶跃函数
def step_function(x):
    return np.floor(x / 9)

# 生成 x 数据
x = np.linspace(0, 100, 500)

# 计算对应的 y 数据
y = step_function(x)

# 绘制阶跃函数
plt.plot(x, y, drawstyle='steps-post')
plt.title('Step Function with Step Size 9')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
