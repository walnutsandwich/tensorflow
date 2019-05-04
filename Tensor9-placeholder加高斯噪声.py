import numpy as np 
import tensorflow as tf 

n_input = 6
n_hidden = 3

def xavier_init(fan_in,fan_out,constant = 1):
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform([fan_in,fan_out],
		minval = low,maxval = high,
		dtype = tf.float32)

x = tf.placeholder(tf.float32,shape=[None,n_input],name="x_input")#设定输入节点数为不限多少维，每一维有n_input个元素，这样计算的时候会把数据一维维输入进去计算
w1 = xavier_init(n_input,n_hidden)
b1 = tf.zeros([n_hidden],dtype = tf.float32)
noise = 0.1*tf.random_normal([n_input,])
hidden = tf.nn.softplus(tf.add(tf.matmul(x+noise,w1),b1))#tf.nn.relu()是自带激活函数

with tf.Session() as sess:
	print('w1的值',sess.run(w1))
	print('b1的值',sess.run(b1))
	print('noise的值',sess.run(noise))
	print('x的值',sess.run(x,feed_dict = {x:[[1,2,3,4,5,6]]}))#输入必须是矩阵形式，不能是向量形式[1,2,3,4,5,6]
	print('hidden',sess.run(hidden,feed_dict = {x:[[1,2,3,4,5,6],[7,8,9,10,11,12]]}))