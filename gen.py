import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，以便结果可复现
np.random.seed(0)

# 生成x值
x = np.linspace(0, 10, 500)

# 生成y值，y=sin(x)并加上正态分布噪声
mu, sigma = 0, 4
y = 10*np.sin(x) + np.random.normal(mu, sigma, 500)

# 将x和y组合成数组
data = np.column_stack((x, y))

# 展示散点图
plt.scatter(x, y)
plt.title('Scatter plot of the generated data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 将数据存储到文件中
np.savetxt('data.txt', data, fmt='%.6f', delimiter=',', header='x,y', comments='')
