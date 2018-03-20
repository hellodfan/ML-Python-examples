#coding:utf8
# 饼状图
import numpy as np
import matplotlib.pyplot as plt

X = [1,2,3,4]

plt.pie(X,	# 要绘制的数据
	labels=['me','you','cat','dog'],  # 输入数据对应的labels
	explode=(0.2,0,0,0),  # 每个部分离中心点的距离
	#shadow=True,		# 饼状图是否有阴影
	autopct='percent:%1.1f%%',	# 每个部分所占的比例标签 支持字符串格式
	pctdistance=0.6,	# 每个部分所占比例的标签离中心点距离
	labeldistance=1.2,	# 每个部分labels离中心点的距离
	radius = 1.2,	# 饼状图的半径
	startangle=90)	# 第一个部分在饼状图上的起始角

plt.axis('equal')  # 防止饼状图被压缩成椭圆
plt.show()
