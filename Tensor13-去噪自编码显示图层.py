import numpy as np 
from sklearn import preprocessing as prep 
#Scikit-learn(sklearn)是机器学习中常用的第三方模块,这里调用数据预处理模块
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data#导入手写识别数据集
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print('训练数据数量',mnist.train.num_examples)
print('测试数据数量',mnist.test.num_examples)
print('验证数据数量',mnist.validation.num_examples)

def xavier_init(fan_in,fan_out,constant = 1):
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in,fan_out),
		minval = low,maxval = high,
		dtype = tf.float32)#tf.random_uniform,返回fan_in*fan_out的矩阵，产生low和high之间均匀分布的值，精度值为32

#去噪自编码的class,为了使这个功能模块化，输入相应的数据的输入输出节点个数，激活函数，优化函数，就可以经过内部过程计算出
class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.5):
	#n_input是输入变量数，n_hidden是隐含层节点数，transfer_function是隐含层激活函数默认为softplus,optimizer是优化器默认是自适应学习率的优化算法tf.train.AdamOptimizer()，scale是高斯噪声系数
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function
		#self.training_scale = scale #原书这里把scale写成了输入变量，但是它是常量所以我把后续该输入删掉跑了也没问题
		#self.scale = tf.placeholder(tf.float32)#默认噪声系数的节点数一维
		self.weights= self._initialize_weights() #利用后面的_initialize_weights函数得到初始的权值参数

		self.x = tf.placeholder(tf.float32,[None,self.n_input])#创建tensorflow输入节点
		self.noise = self.x + scale*tf.random_normal([n_input,])
		self.hidden = self.transfer(tf.add(tf.matmul(
			self.x + scale*tf.random_normal([n_input,]), #将输入x加上噪声,tf.random_normal((n_input,))从正态分布中随机取出数值
			self.weights['w1']),self.weights['b1']))#将加了噪声后的输入与隐藏层权重w1相乘再加上偏置b1，最后用激活函数self.transfer=tf.nn.softplus得到隐藏层数据

		self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])#重建得输出层，直接将隐含层输出self.hidden乘上输出层权重w2,再加上偏置b2，不需要激活函数

		self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(
			self.reconstruction,self.x),2.0))       #定义损失函数，用平方误差(),tf.pow为计算幂指数，tf.subtract为计算减法，tf.reduce_sum为按照维度求和

		self.optimizer = optimizer.minimize(self.cost) #自适应学习率算法，设置.minimize()是最小化的目标变量，minimize解释https://www.jianshu.com/p/72948cce955f，三种常用学习方法https://www.jianshu.com/p/200f3c4336a3

		init = tf.global_variables_initializer() #定义初始化全部参数
		self.sess = tf.Session() #运行计算图
		self.sess.run(init)#运行初始化
#前面是定义各种需要得到的参数，后面函数基本上就是填数据运行图来计算参数的过程
	def _initialize_weights(self):
			all_weights = dict() #设定一个字典来存储权值和偏置，然后向里面添加属性和参数
			all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))#第一层权值需要用前面定义的xavier初始化
			all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype = tf.float32))
			all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype = tf.float32))
			all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
			return all_weights

	def partial_fit(self,X):        #把数据X输入进去，计算损失函数并优化权值偏置参数
			cost,opt = self.sess.run((self.cost,self.optimizer),
			feed_dict = {self.x:X})#运行cost和optimizer
			return cost

	def calc_total_cost(self,X):     #定义一个只求损失cost的函数，自编码器训练完毕后对全部数据进行测评，不触发训练
			return self.sess.run(self.cost,feed_dict = {self.x:X})

	def transform(self,X):           #返回自编码器隐藏层结果，表示数据中的高阶特征
			return self.sess.run(self.hidden,feed_dict = {self.x:X,})

	def generate(self,hidden = None):    #将隐含层输出作为输入，复原出原始数据与transform函数一起构成自编码器的两部分
			if hidden is None:  #hidden = None是一个防止报错的机制，如果没有传参数，就默认为None状态，如果传了参数，则取所传的参数 
				hidden = np.random.normal(size = self.weights['b1']) #默认为None状态时，设定同weights['b1']形状的随机赋值
			return self.sess.run(self.reconstruction,
				feed_dict = {self.hidden:hidden})

	def reconstruct(self,X):           #完整运行编码解码过程，即包括transform和generate两部分
			return self.sess.run(self.reconstruction,feed_dict = {self.x:X})

	def getWeights(self):              #获取全部权重
			return self.sess.run(self.weights)

	def getnoise(self,X):				#获取隐含层偏置b1
			return self.sess.run(self.noise,feed_dict = {self.x:X})

def standard_scale(X_train,X_test):  #将数据标准化处理，让数据变成0均值，标准差为1，方法是先减去均值，再除以标准差。
	preprocessor = prep.StandardScaler().fit(X_train) #StandardScaler是sklearn.preprossing里一个用来将数据进行归一化和标准化的类，能够调用fit方法
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train,X_test

def get_random_block_from_data(data,batch_size): #定义一个随机顺序获取一块block数据的函数，
	start_index = np.random.randint(0,len(data) - batch_size) #取一个0到len(data)-batch_size之间的随机整数
	sample_batch = data[start_index:(start_index + batch_size)]
	num = list(range(start_index,(start_index + batch_size)))
	data = np.delete(data,num,axis=0) #利用numpy.delete删除抽出来的数据
	return sample_batch,data #实现不放回的抽样

X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)#读取训练和测试数据，无监督学习不需要标签数据
n_samples = int(mnist.train.num_examples) #训练样本个数，取整数
training_epochs = 20 #设置最大训练轮数为20，每隔一轮显示一次损失
batch_size = 128
display_step = 1 #设置每隔多少轮显示一次损失

#接下来创建一个AGN自编码器的实例，定义模型输入节点数，隐藏层节点数，激活函数，优化器和学习率，噪声系数
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,n_hidden = 196,transfer_function = tf.nn.softplus,optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),scale = 0.5)

#下面开始训练过程,不放回的抽样
for epoch in range(training_epochs):
	train_data = X_train
	avg_cost = 0                    #将平均损失初始化为0
	total_batch = int(n_samples/batch_size)        #总样本数除以每一轮训练的个数取整得到训练轮数
	for i in range(total_batch):    #对于每一轮训练
		batch_xs,train_data = get_random_block_from_data(train_data,batch_size)#随机按顺序获取一块数据
		cost = autoencoder.partial_fit(batch_xs) #cost是一个batch数据的均方误差
		avg_cost += cost/n_samples*batch_size #把n_samples/batch_size份个batch的cost加起来就是n_samples总的cost
	if epoch % display_step == 0:
		print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
		#每隔display_step训练次数就显示一次，第几轮训练（'%04d'%k显示4位整不足4位补零），损失（小数点后9位）format参考https://www.cnblogs.com/wushuaishuai/p/7687728.html

print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))
#autoencoder.pltTwo()
#以下是测试数据的图形化显示
def showdata(data):
	q = int(pow(len(data),0.5))
	one_pic_arr = np.reshape(data,(q,q))
	pic_matrix = np.matrix(one_pic_arr, dtype="float")
	plt.imshow(pic_matrix)#用matplotlib命令显示出来

endweights = autoencoder.getWeights()
print('训练后权值：',endweights)
test_noise = autoencoder.getnoise(X_test)
test_hidden = autoencoder.transform(X_test)
test_recon = autoencoder.reconstruct(X_test)

for i in range(len(X_test)):
	print('测试样本-%04d:'%(i+1),X_test[i])
	print('测试样本加噪-%04d:'%(i+1),test_noise[i])
	print('测试样本隐含层-%04d:'%(i+1),test_hidden[i])
	print('测试样本还原-%04d:'%(i+1),test_recon[i])
	plt.subplots_adjust(wspace =0, hspace =0.3)
	plt.subplot(2,2,1).set_title('X_test')
	showdata(X_test[i])
	plt.subplot(2,2,2).set_title('X_test+noise')
	showdata(test_noise[i])
	plt.subplot(2,2,3).set_title('hidden')
	showdata(test_hidden[i])
	plt.subplot(2,2,4).set_title('reconstruction')
	showdata(test_recon[i])
	plt.show()

