#coding:utf8
# 绘制子图
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure()

# 创建一个3*3大小的子图格，该子图的起始位置为(0,0) 列跨度为3 行跨度为1
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')

# 创建一个3*3大小的子图格，该子图的起始位置为(1,0) 列跨度为2
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,)
ax3 = plt.subplot2grid((3,3),(1,2),rowspan=2,)
ax4 = plt.subplot2grid((3,3),(2,0),)
ax5 = plt.subplot2grid((3,3),(2,1),)

plt.show()




