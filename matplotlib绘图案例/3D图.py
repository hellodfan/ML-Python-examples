#coding:utf8
# 绘制3D图像
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

# 创建数据
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X,Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 将数据绘制到坐标轴上 颜色映射为rainbow
suf = ax.plot_surface(X, Y, Z, 
		rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# 绘制等高线，选择的绘制方向是z轴视角
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')

fig.colorbar(suf,shrink=0.5)  # 添加颜色bar

ax.set_zlim(-2, 2)


plt.show()











