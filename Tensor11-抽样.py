import random
import numpy as np
A = list(range(30))#或写成A=[i for i in range(100)]，产生一串顺序数字
batch_size = 3

def get_random_block(data,batch_size): #定义一个随机顺序获取一块block数据的函数，
	start_index = random.randint(0,len(data) - batch_size) #取一个0到len(data)-batch_size之间的随机整数
	sample = data[start_index:(start_index + batch_size)]
	data[start_index:(start_index + batch_size)]=[]
	return sample #按顺序取一个的batch_size的数据

for i in range(int(len(A)/batch_size)):
	batch_A = get_random_block(A,batch_size)
	print(batch_A,"%",A)

#==对于矩阵抽样不放回==============================
B = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18]]

def get_matrix_block(data,batch_size): 
	start_index = random.randint(0,len(data) - batch_size)
	sample = data[start_index:(start_index + batch_size)]
	num = list(range(start_index,(start_index + batch_size)))
	# print(start_index,start_index + batch_size)
	# print(num)
	data = np.delete(data,num,axis=0)
	return sample,data 

for i in range(int(len(B)/batch_size)):
	batch_B,B = get_matrix_block(B,batch_size) 
	print(batch_B,"%",B)