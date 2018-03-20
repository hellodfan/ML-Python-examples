# coding:utf8
# Keras 快速开始 在MNIST上构建一个简单的神经网络
# Author: DFan
from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import mnist
from keras.utils import to_categorical

# 下载并载入MNIST数据集
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()

# reshape数据集，并归一化
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建网络模型
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28*28,)))
network.add(Dense(10,activation='softmax'))


network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 训练并评估模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:',test_acc)