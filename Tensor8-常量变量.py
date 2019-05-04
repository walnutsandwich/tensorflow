import tensorflow as tf 
#c常见张量类型参考https://www.cnblogs.com/tensorflownews/p/8671397.html
m1 = tf.constant([3,5])
m2 = tf.constant([2,4])
result = tf.add(m1,m2,name='mul')
print(result)#上述构建模型，不运行的话只会输出Tensor

#运行输出方法一
sess = tf.Session()
print(sess.run(result))
sess.close
#运行输出方法二
with tf.Session() as sess: #推荐此种写法可以字典关闭会话，无需调用close释放资源
	res = sess.run(result)
print(res)
#运行输出方法三
sess = tf.InteractiveSession()
print(result.eval())

#常量初始化一个零矩阵
a = tf.constant([2.0,3.0],name='a',shape=(2,0),dtype="float64",verify_shape="false")#常量有五个参数
b = tf.zeros([2,2],tf.float32)
with tf.Session() as sess:
	print(sess.run(a))
	print(sess.run(b))
#创建和初始化随机张量:[2,3]是形状、mean是正态分布的均值、stddev是正态分布标准差
random1 = tf.random_normal([2,3],mean=-1,stddev=4,dtype=tf.float32,seed=None,name='rnum')
random2 = tf.random_uniform((3,3),minval=-1,maxval=1,dtype=tf.float32)#均匀分布变量
with tf.Session() as sess:
	print(sess.run(random1))
	print(sess.run(random2))

#变量的创建，第一个参数表示形状,使用前一定要初始化
A = tf.Variable(3,name="number")
B = tf.Variable([1,3],name="vector")
C = tf.Variable([[0,1],[2,3]],name="matrix")#创建了变量初始值为矩阵[[0,1],[2,3]]
D = tf.Variable(tf.zeros([100]),name="zero")#创建了初始值为0的零矩阵
E = tf.Variable(tf.random_normal([2,3],mean=1,stddev=2,dtype=tf.float32))
#变量的初始化一（初始化全部变量）
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
#变量的初始化二（初始化单个变量）
init_var = tf.Variable(tf.zeros([2,5]))#初始化一个2*5的零矩阵变量
with tf.Session() as sess:
	sess.run(init_var.initializer)
#变量的保存(存储器方法)
init = tf.global_variables_initializer()#要对变量初始化才能保存
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	save_path = saver.save(sess,"Tensorflow/save.ckpt")#设置存储路径，保存内容是从变量名到tensor值的映射，可以用saver.restore(sess,文件位置)方法获取

#一个变量常量差别的小测试
import numpy as np 
n_input = 6
n_hidden = 3
def xavier_init(fan_in,fan_out,constant = 1):
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.Variable(tf.random_uniform((fan_in,fan_out),minval = low,maxval = high,dtype = tf.float32))
#定义变量和常量的区别就在于，变量只有在第一次初始化时会按照规则初始化，后续调用该变量不会改变其数值
#可以测试改成tf.random_uniform((fan_in,fan_out),minval = low,maxval = high,dtype = tf.float32)
w1 = xavier_init(n_input,n_hidden) #w1是变量时，run(w1)就是读取其数值
w2 = w1*10
init = tf.global_variables_initializer()#变量只初始化这一次
sess.run(init)
print(sess.run(w1))
print(sess.run(w2))
print(sess.run(w1))

#数据初始化容器palceholder
c = tf.placeholder(tf.float32,shape=[2],name=None)
d = tf.constant([6,4],tf.float32)
e = tf.add(c,d)
with tf.Session() as sess:
	print(sess.run(e,feed_dict={c:[10,10]}))
#关于placeholder的节点形状定义
input1 = tf.placeholder(tf.float32,shape=[1, 2],name="input-1")#定义节点为一维每个维度有两元素的矩阵，即一行两个元素，或者一列两行
input2 = tf.placeholder(tf.float32,shape=[2, 1],name="input-2")#定义节点为二维每个维度有一个元素的矩阵，即行数为2，列数为1
output1 = tf.multiply(input1,input2)#两个矩阵中对应元素各自相乘,即不求和的点乘
output2 = tf.matmul(input1, input2)#将矩阵a乘以矩阵b，生成a*b,叉乘
with tf.Session() as sess:
	print (sess.run(output1, feed_dict = {input1:[[1,2]], input2:[[3],[4]]}))#结果[[3. 6.][4. 8.]]
	print (sess.run(output2, feed_dict = {input1:[[1,2]], input2:[[3],[4]]}))#结果[[11]]

#fetch的用法，run方法可以同时计算多个tensor值
f = tf.constant(5)
g = tf.constant(6)
h = tf.constant(4)
add = tf.add(g,h)
mul = tf.multiply(f,add)#相乘
with tf.Session() as sess:
	result = sess.run([mul,add])
	print(result)