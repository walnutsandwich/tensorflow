import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
hello = tf.constant('hello,tensorf') #创建一个常数张量,传入list或者数值来填充
sess = tf.Session() #Session()是 Tensorflow 控制和输出文件的执行的语句.
print(sess.run(hello)) #输出'hello.tensorf'
#============================================
import tensorflow
a = tensorflow.constant(10)
b = tensorflow.constant(22)
sess = tensorflow.Session()
print(sess.run(a+b))