#coding:utf8
# 等高线
import numpy as np
import matplotlib.pyplot as plt

def calcHigh(x, y):
	'''  计算高度值	'''
	return (1-x/2+ x**5 + y**3 ) * np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x,y) # 将xy放置网格中


# 使用contourf填充等高线内颜色 
# 颜色映射使用hot 使用8表示分为10部分  0表示2部分
plt.contourf(X, Y, calcHigh(X, Y), 8, alpha=0.75, cmap=plt.cm.hot) 
#plt.contourf(X, Y, calcHigh(X, Y), 8, alpha=0.75, cmap=plt.cm.hot) # 颜色映射使用cold

# 使用contour函数绘制等高线
C = plt.contour(X, Y, calcHigh(X, Y), 8, colors='k', linewidth=0.5) 

# 使用clabel对等高线做label描述
plt.clabel(C, inline=True, fontsize=10) 

#plt.xticks(()) # 隐藏刻度
#plt.yticks(())

plt.show()
