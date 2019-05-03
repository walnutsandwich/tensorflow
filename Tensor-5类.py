class person():
	def __init__(self,name,gender,birth,**kw):#类里包括多个函数时，先用__init__方法初始化全局变量，双星号变量**kw代表一个字典
		self.name = name  #self代表类本身，相当于外部调用时person().name
		self.gender = gender
		self.birth = birth
		for k,w in kw.items():#items将字典以列表形式返回
			setattr(self,k,w)#setattr() 函数用于设定第一个属性参数的值设为第二个参数
	def sayhi(self):
		print('my name is',self.name)
	def sayha(self):
		w = 'my name is '+self.name
		return(w)

xiaoming = person('Xiaoming','Male','1991',job='student',tel='159157')#此处job='student',tel='159157'即字典形式的变量
xiaohong = person('Xiaohong','Female','1992')

print(xiaoming.name)
print(xiaoming.birth)
print(xiaoming.job)#前面需要setattr赋值给该属性变量后才能将字典里的属性读取出来
print(xiaoming.tel)
xiaoming.sayhi()
print(xiaoming.sayhi())
print(xiaoming.sayha())
#如果定义class函数里有print 最后的输出时也print，结果后面就会跟着一个none