import numpy as np
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽掉系统警告
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

sess = tf.InteractiveSession()#构建计算图并在运行计算图的时候能对某些节点的运算进行修改
x = tf.placeholder(tf.float32,[None,784])#先构建存储输入数据的节点大小，之后再向其中喂入数据
W = tf.Variable(tf.zeros([784,10])) #通过创建Variable实例向graph中添加图变量。
b = tf.Variable(tf.zeros([10])) #创建了一个权值矩阵和一个偏置向量
y = tf.nn.softmax(tf.matmul(x,W)+b)#tf.matmul命令即用矩阵x乘以W,tf.nn.softmax为都计算一个exp再标准化
#接下来定义好信息熵作为损失函数（值越小表示分类误差越小），即可自动完成前向传播和反向梯度下降来自动学习
y_ = tf.placeholder(tf.float32,[None,10])#构建存储输出数据的节点
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#tf.reduce_mean用于计算张量tensor沿着某一维度求均值reduction_indices值为1表示函数的处理维度为横向，值为0表按纵向求均值
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#调用tf.train.GradientDescentOptimizer表示用随机梯度下降法来优化权值和偏置，并设置好损失函数
tf.global_variables_initializer().run()#初始化全局变量并开始启动计算图
for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)#next_batch命令为从mnist.train里随机提取100个样本来训练，循环1000次
	train_step.run({x:batch_xs,y_:batch_ys})#每次喂进去100个随机样本并运行train_step
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#tf.argmax(input矩阵,axis)为每一列(0)/行(1)的元素，返回其最大值索引，比较两组向量得出是否准确的1/0（正确/错误）数组
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#tf.cast把数据格式转成float32,然后求均值
test = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})#eval()表示输入测试样本启动计算并且最终得到accuracy值,
#也可以写成test = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
print('分类测试样本准确率：',test)#打印出来

for i in range(0, len(mnist.test.images)):
  result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
  if not result:#如果预测值与标签值不一致
    print('预测的值是：',sess.run(y, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
    print('实际的值是：',sess.run(y_,feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
    one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))#这里把onehot向量重新组成矩阵
    pic_matrix = np.matrix(one_pic_arr, dtype="float")
    plt.imshow(pic_matrix)#用matplotlib命令显示出来
    plt.show()
    continue