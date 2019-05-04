#建议查看StandardScaler说明https://scikit-learn.org/stable/modules/preprocessing.html
#参考https://www.cnblogs.com/cola-1998/p/10218276.html
#参考https://blog.csdn.net/weixin_39175124/article/details/79463993
#参考https://blog.csdn.net/onthewaygogoing/article/details/79871559
from sklearn.preprocessing import StandardScaler
data = [[-1,0],[1,0],[1,1],[1,1]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean_)#求均值
print(scaler.scale_)#求标准差
#=====================================================================
import numpy as np 
import warnings
warnings.filterwarnings("ignore")#有个数据格式的警告，忽略掉
x_train = np.arange(10).reshape(5,2)
x_test = np.arange(3,7).reshape(2,2)
y = [1,0,0,0,1]
ss = StandardScaler(copy=True, with_mean=True, with_std=True) #调用StandardScaler类，此处参数为默认，copy 如果为false,就会用归一化的值替代原来的值，with_mean 在处理sparse CSR或者 CSC matrices 一定要设置False不然会超内存
print(x_train,x_test)
z = ss.fit_transform(x_train)#等同于ss.fit(x_train).transform(x_train)即先拟合x_train数据，然后将其标准化为均值为0、标准差为1的数据

w = ss.fit(x_train)#运行fit方法拟合得到均值和方差等参数,fit的第二个参数为y=标签数据，默认为None
print(ss.n_samples_seen_,ss.mean_,ss.var_,ss.scale_)
#参数解释：n_samples_seen_样本数量，mean_每个特征的均值，var_每个特征方差，scale_每个特征标准差
x_train = w.transform(x_train)
x_test = w.transform(x_test)#用训练集的拟合参数来标准化测试集，机器学习中有很多假设，这里假设了训练集的样本采样足够充分

print(z)
print(x_train,x_test)#转换后的训练和测试数据
#如果原始数据的分布 不 接近于一般正态分布，则标准化的效果会不好
print(ss.get_params(deep=True))#返回StandardScaler对象的设置参数
print(ss.inverse_transform(x_test,copy=True))#StandardScaler()会保存标准化参数并且逆向转换