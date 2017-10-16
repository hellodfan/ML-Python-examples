#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,5,10)  
y1= x
y2 =x**2 -1

plt.figure()	
l1, = plt.plot(x, y1)
l2, = plt.plot(x, y2, label='you') 

plt.legend(handles=[l1,l2], labels=['me','you'], title='I am legend',
	loc='lower right',ncol=2, fontsize='large',facecolor='r',edgecolor='b')

plt.show()

