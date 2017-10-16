#coding:utf8
# 设置刻度格式
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,5,10)  
y1= x
y2 =x**2 -2

plt.figure()	
l1, = plt.plot(x, y1)
l2, = plt.plot(x, y2, label='you',linewidth=10) 

ax  = plt.gca()

plt.xlim((-2, 2)) # 设置x轴显示的范围
plt.ylim((-2, 2))

ax.spines['right'].set_color('none')  # 设置右边的坐标轴不显示
ax.spines['top'].set_color('none')  # 设置上边的坐标轴不显示

ax.spines['bottom'].set_position(('data',-1))  # 将x轴绑定在y轴的-1的位置  
ax.spines['left'].set_position(('data',0))   # 将y轴绑定在x轴的0的位置

for label in ax.get_xticklabels() +ax.get_yticklabels():
	label.set_fontsize(12)
	label.set_bbox(dict(facecolor='r',edgecolor='None',alpha=0.2))

plt.show()

