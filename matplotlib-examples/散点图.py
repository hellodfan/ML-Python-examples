#coding:utf8
# 散点图
import numpy as np
import matplotlib.pyplot as plt

n = 1024
X = np.random.normal(0, 1, n) # 散点数据
Y = np.random.normal(0, 1, n)

T = np.arctan2(Y,X) # 设置一个点对应的color值，只是为了好看

plt.scatter(X, Y, s=75, c=T, alpha=0.5) # 绘制散点图，并设置对应的颜色关系

plt.xlim(-1.5,1.5)  # 设置x轴显示刻度范围
plt.ylim(-1.5,1.5)

plt.xticks(())  # 设置不显示刻度
plt.yticks(())
plt.show()
