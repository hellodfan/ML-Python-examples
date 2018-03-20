#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2,10)  
y1= x
y2 =x**2

plt.figure()	
l1, = plt.plot(x, y1)
l2, = plt.plot(x, y2, label='you') 


plt.ylim(0,2)
plt.xlim(0,2)

# 找到需要标注的点
y0 = x0 = 1
plt.scatter(x0, y0, s=50, color='b') # 设置大小为50，蓝色
plt.plot([1,1],[0,1],'k--')  # 绘制标注点到x轴之间的虚线
plt.plot([0,1],[1,1],'k--')  # 绘制标注点到y轴之间的虚线


# 使用annotate进行注解描述
plt.annotate(r'$ k_i=%s $'% y0,  # 指定注解内容 可以使用Latex
			xy=(x0, y0), xycoords='data', xytext=(+30, -30), # xy为基准坐标  coords设置偏移坐标
			textcoords='offset points', fontsize=16, # 设置字体大小
			arrowprops=dict(arrowstyle='->',	# 设置箭头类型
			connectionstyle='arc3,rad=.2'))


# 使用text指定注解
x1 = 0.25
y1 = 1.5
plt.text(x1, y1, 		# 设置基准坐标
		 r'$function : y={x_i}^2 $',	# 设置显示内容，可以使用Latex
		 fontdict={'size':18, 'color':'r'})	# 设置字体


plt.show()

