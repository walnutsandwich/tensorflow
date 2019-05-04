import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

train_x = np.linspace(-1,1,100)#返回均匀分布的样本
train_y = 2*train_x + np.random.randn(*train_x.shape)*0.33 + 10
#numpy.random.randn以给定的形状创建符合标准正太分布的数组，因为x.shape返回的是元组，所以要加*号解包
X = tf.placeholder('float')
Y = tf.placeholder('float')
w = tf.Variable(0.0,name="weight")
b = tf.Variable(0.0,name="bias")
loss = tf.reduce_sum(tf.square(Y- X*w - b)) #定义损失函数tf.square是对每一个元素求平方，这里损失也可以不求和
op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#这里可以尝试其它几种优化器的结果和调参,用tf.train.AdamOptimizer(0.1)时学习率需要调高一点
#minimize()函数的作用仅仅是朝使每个样本loss变小的方向去调参，并没有自动迭代的功能
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	epoch = 1
	for i in range(10): #这里循环迭代十次
		for (x,y) in zip(train_x,train_y): #zip()将对象中对应位置的元素打包成一个个元组，然后返回由这些元组组成的列表
			_,w_value,b_value,loss_value = sess.run([op,w,b,loss],feed_dict={X:x,Y:y})#_”单下划线，意味着该方法或属性不应该去调用
			print("Epoch:{},w:{},b:{},loss:{}".format(epoch,w_value,b_value,loss_value))
			epoch += 1
plt.plot(train_x,train_y,"+")
plt.plot(train_x,train_x.dot(w_value)+b_value)
plt.show()