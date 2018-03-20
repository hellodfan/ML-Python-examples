#coding:utf8
# 柱状图
import numpy as np
import matplotlib.pyplot as plt

# 产生数据
n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = -((1-X/float(n)) * np.random.uniform(0.5, 1.0, n))

# 使用bar函数绘制多个柱状图，并设置不同的颜色
plt.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, Y2, facecolor='#ff9999', edgecolor='white')

# 为所有柱状图标注图解 这里使用text来绘制(感觉text方便点)
for x,y in zip(X,Y1):   # 使用zip同时传递X，Y1到x,y上
	plt.text(x, y+0.05, '%.2f'%y, ha='center',va='bottom') # ha: horizontal alignment

for x,y in zip(X,Y2):   
	plt.text(x, y-0.05, '%.2f'%y, ha='center',va='top') # ha: horizontal alignment


plt.xlim(-0.5,n)  # 设置x轴显示刻度范围
plt.ylim(-1.25,1.25)

plt.xticks(())  # 设置不显示刻度
plt.yticks(())
plt.show()
