import numpy as np
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#mnist = tf.keras.datasets.mnist
#(x_train,y_train),(x_test,y_test)= mnist.load_data()
#下载mnist数据并且在当前目录下创建目录并且MNIST_data/并且生成文件,one_hot=True,表示预测目标值以及经过One-Hot编码
#由于网络连接失败，我们直接把下载好的mnist
x_train = mnist.train.images
k = mnist.train.labels
y_train = np.asarray(mnist.train.labels,dtype=np.int32)#将列表转换为数组，数据类型转为np.int32整数十位数范围
x_test = mnist.test.images
y_test = np.asarray(mnist.test.labels,dtype=np.int32)
x_val = mnist.validation.images
y_val = np.asarray(mnist.validation.labels,dtype=np.int32)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)
print('训练数据数量',mnist.train.num_examples)
print('测试数据数量',mnist.test.num_examples)
print('验证数据数量',mnist.validation.num_examples)
print(k.shape)
#手写识别数据样本为28*28个点构成的特征，标签为一个one-hot向量表示0到10，
#全部数据总共分三个集合：训练数据、测试数据、验证数据