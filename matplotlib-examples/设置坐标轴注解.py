#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,7,50)  
y1= x

plt.figure()	
plt.plot(x, y1)

plt.xlim((-1, 2)) # 设置x轴显示的范围
plt.ylim((-2, 3))

plt.xlabel('I am x !') # 设置x轴的label
plt.ylabel('y am I !')

new_ticks = np.linspace(-1, 2 , 10)  
print('new_ticks:',new_ticks)
plt.xticks(new_ticks)		# 设置x轴的刻度
plt.yticks([-2, -1, 0, 1, 2, 3],  	# 设置y轴的刻度 
	['level0','level1',			# 可以看到坐标轴的刻度可以设置为字符串 
	r'$ \ mid$',			    # 这里设置的对应的字符串规则就是常见Latex格式
	r'$ \ \alpha $',		 	# 这样的规则很适合在论文表现数据
	r'$ \ \pi $',
	r'$ \frac{\alpha} {\theta}  $']) 	# 设置一个分子式

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')  # 设置右边的坐标轴不显示
ax.spines['top'].set_color('none')  # 设置上边的坐标轴不显示
ax.xaxis.set_ticks_position('bottom') # 设置当前坐标的默认x轴
ax.yaxis.set_ticks_position('left') # 设置当前坐标的默认y轴

ax.spines['bottom'].set_position(('data',-1))  # 将x轴绑定在y轴的-1的位置  data,outward,axes
ax.spines['left'].set_position(('data',0))   # 将y轴绑定在x轴的0的位置

plt.show()


# 创建一个fig对象，在下一个fig对象创建前管理所有绘图资源
#plt.figure(figsize=(10,10),dpi=100,facecolor='g',edgecolor='r')	
#plt.plot(x, y1)  
#plt.show()

# 创建新fig对象，管理下面的操作的绘图资源
#plt.figure(num=3, figsize=(8, 5))	
#plt.plot(x, y1)
#plt.plot(x, y1,x, y2, color='red',linewidth=3.0,linestyle='--',antialiased=False)

#plt.show()