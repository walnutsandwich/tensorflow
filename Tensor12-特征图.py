import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
n_input = 9
n_hidden = 4
scale = 0.1
x = tf.placeholder(tf.float32,[None,n_input])
X = np.random.randn(9,9) #随机产生一个9行9列的训练样本
X_test = np.random.randn(3,9) #随机产生一个3行9列的测试样本

def xavier_init(fan_in,fan_out,constant = 1):
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in,fan_out),
		minval = low,maxval = high,
		dtype = tf.float32)#tf.random_uniform,返回fan_in*fan_out的矩阵，产生low和high之间均匀分布的值，精度值为32

def _initialize_weights():
	all_weights = dict() #设定一个字典来存储权值和偏置，然后向里面添加属性和参数
	all_weights['w1'] = tf.Variable(xavier_init(n_input,n_hidden))#第一层权值需要用前面定义的xavier初始化
	all_weights['b1'] = tf.Variable(tf.zeros([n_hidden],dtype = tf.float32))
	all_weights['w2'] = tf.Variable(tf.ones([n_hidden,n_input],dtype = tf.float32))
	all_weights['b2'] = tf.Variable(tf.zeros([n_input],dtype = tf.float32))
	return all_weights

def partial_fit(X):        #把数据X输入进去，计算损失函数并优化权值偏置参数
	cost,opt = sess.run((lost,optimizer),
	feed_dict = {x:X})#运行lost和optimizer
	return cost

w1 = xavier_init(n_input,n_hidden)
weights = _initialize_weights()
hidden = tf.nn.softplus(tf.add(tf.matmul(
			x + scale*tf.random_normal([n_input,]), #将输入x加上噪声,tf.random_normal((n_input,))从正态分布中随机取出数值
			weights['w1']),weights['b1']))#将加了噪声后的输入与隐藏层权重w1相乘再加上偏置b1，最后用激活函数self.transfer=tf.nn.softplus得到隐藏层数据

reconstruction = tf.add(tf.matmul(hidden,weights['w2']),weights['b2'])
lost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(reconstruction,x),2.0))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(lost) 

init = tf.global_variables_initializer()#初始化一定要放在w2 = _initialize_weights()定义变量的后面

saver = tf.train.Saver()#保存
sess = tf.InteractiveSession()
sess.run(init)
print(sess.run(w1))
print('训练前权值：',sess.run(weights))
print('隐藏层：',sess.run(hidden,feed_dict = {x:X}))
print('恢复后：',sess.run(reconstruction,feed_dict = {x:X}))
print('训练前权值：',sess.run(weights))
loss = partial_fit(X)#可以看到训练后权值会改变
print(loss,'训练后权值：',sess.run(weights))

test_noise = x + scale*tf.random_normal([n_input,])
test_hidden = tf.add(tf.matmul(x + scale*tf.random_normal([n_input,]), weights['w1']),weights['b1'])
test_reconstruction = tf.add(tf.matmul(test_hidden,weights['w2']),weights['b2'])

def showdata(data):
	q = int(pow(len(data),0.5))
	one_pic_arr = np.reshape(data,(q,q))
	pic_matrix = np.matrix(one_pic_arr, dtype="float")
	plt.imshow(pic_matrix)#用matplotlib命令显示出来

test_no,test_hi,test_re = sess.run((test_noise,test_hidden,test_reconstruction),feed_dict={x:X_test})

for i in range(len(X_test)):
	print('测试样本-%02d:'%(i+1),X_test[i])
	print('测试样本加噪-%02d:'%(i+1),test_no[i])
	print('测试样本加噪-%02d:'%(i+1),test_hi[i])
	print('去噪恢复后-%02d:'%(i+1),test_re[i])
	plt.subplots_adjust(wspace =0, hspace =0.3)#wspace是调间距宽度，hspace是调间距高度
	#plt.tight_layout(8,rect=[0.5, 0, 1, 1])是设置坐标刻度和网格顶部底部范围
	plt.subplot(2,2,1).set_title('X_test')
	showdata(X_test[i])
	plt.subplot(2,2,2).set_title('X_test+noise')
	showdata(test_no[i])
	plt.subplot(2,2,3).set_title('hidden')
	showdata(test_hi[i])
	plt.subplot(2,2,4).set_title('reconstruction')
	showdata(test_re[i])
	plt.show()
	continue