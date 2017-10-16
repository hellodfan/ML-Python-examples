#coding:utf8
# 绘制子图
import numpy as np
import matplotlib.pyplot as plt

plt.figure()

# 绘制一个子图，一共分为row=2,col=2 ,该子图占第1个位置
plt.subplot(2, 2, 1)  
plt.plot([0, 1], [0, 1])

# 绘制一个子图，一共分为row=2,col=2 ,该子图占第2个位置
plt.subplot(2, 2, 2)
plt.plot([0, 1], [2, 2])

plt.subplot(2, 2, 3)
plt.plot([1, 1], [2, 3])

plt.subplot(2, 2, 4)
plt.plot([1, 2], [2, 1])

plt.show()




