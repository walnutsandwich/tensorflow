import numpy as np 
import tensorflow as tf 

def xavier_init(fan_in,fan_out,constant = 1):
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform([fan_in,fan_out],
		minval = low,maxval = high,
		dtype = tf.float32)
#tf.random_uniform,返回fan_in*fan_out的矩阵，产生low和high之间均匀分布的值，精度值为32
w1 = xavier_init(6,6)
with tf.Session() as sess:
	print(sess.run(w1))

#=以上是常量形式==以下是变量形式=======================
v1 = tf.Variable(w1)#创建变量后要初始化和保存后才能运行计算
v2 = tf.Variable([1,3],name="vector")#创建一个初始值为[1,3]的变量，创建变量需要一个任意类型和值的tensor作为初始值
init = tf.global_variables_initializer()#初始化全局变量，后面要先run该初始化，再run变量
saver = tf.train.Saver()#保存
with tf.Session() as sess:
	sess.run(init)
	w2 = sess.run(v1)
	print(sess.run(v2))
print(w2)