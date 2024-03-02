import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，以便结果可复现
np.random.seed(0)

# 生成x值
x = np.linspace(-10, 20, 5000)

# 生成y值，y=sin(x)并加上正态分布噪声
mu, sigma = 0, 0.1
# y = 3 * np.cos(x) + np.random.normal(mu, sigma, 60)
y = -x+2+ np.random.normal(mu, sigma, 5000)

# 将x和y组合成数组
data = np.column_stack((x, y))

# 展示散点图
plt.scatter(x, y)
plt.title('Scatter plot of the generated data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 将数据存储到文件中
np.savetxt('data.txt', data, fmt='%.6f', delimiter=',', comments='')
