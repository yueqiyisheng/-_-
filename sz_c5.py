# 机器学习实战 学习记录
# Chapter 5 Logistic 回归
# coding='UTF-8'
'''
使用sigmoid函数进行二分类；
z=wx
w为参数，需要基于训练集使用最优化方法（如梯度上升法）寻参
'''
'''
梯度上升法伪代码：

每个回归系数（wi）初始化为1
重复r次：
	计算整个数据集的梯度
	使用α×gradient更新回归系数的向量 w:=w+α×gradient
	返回回归系数
'''

# import numpy as np
from numpy import * 
# 不需要在每个numpy中的函数，如mat前面加上np.，同时不用import math（math.exp时似乎不能处理数组）
# numpy 有指数函数，np.exp() 是可以直接处理数组的！！！ 但是math.exp()是不行的

def loaddataset(): #从文本文件中读取数据 list
	dataset = [];labelset = []
	filename = r'.\data\testSet.txt'
	datalines = open(filename).readlines()
	for line in datalines:
		item = line.strip().split('\t') # 读出来的是str，需要转换成相应的数值型！！
		dataset.append([1.0,float(item[0]),float(item[1])]) # 注意w参数中会有一项常数，所以这里需要在前面加上一个1
		labelset.append(int(item[2]))
	return dataset,labelset

def sigmoid(z):
	# import math
	return 1.0/(1+ exp(-z))
	
# 梯度上升法
# 输入：数据集（list）和label（list）
# 求梯度需要用到矩阵运算，所以需要将list转换成numpy中的矩阵
def gradascent(dataset,labelset):
	datamat = mat(dataset) # 训练样本数 × 特征数+1
	labelmat = mat(labelset).transpose()
	m,n = shape(datamat)
	alpha = 0.001 #移动步长 自己设置合适的值？
	maxcycle = 500 # 迭代次数
	weight = ones((n,1)) # 从shape得到的共有n-1个特征，再加上一个1
	for i in range(maxcycle):
		h = sigmoid(datamat*weight) # !矩阵相乘 得到 m×1 结果 注意一下均是矩阵运算
		error = labelmat - h
		weight = weight + alpha * datamat.transpose()* error # 目的是让误差最小化【注意推导！！
	return weight  # 初始化时w还是array，array与matrix运算，得到的是matrix ！！！
	
# import sz_c5
# dataset,labelset = sz_c5.loaddataset()
# w = sz_c5.gradascent(dataset,labelset)

# 上面的函数已经得到参数weight，下面将边界可视化
'''
matrix 和 array是不同的
matrix.getA() 等同于 np.asarray(self) 将矩阵转换为 ndarray
ndarray可以是多维的，运算是元素级别的，即*是元素的相乘，矩阵乘法必须用dot()进行
matrix必须是二维的，运算是矩阵级别的，即*是矩阵乘法，multiply()实现点乘
'''
def plotbestfit(wei): #输入的是weight的矩阵
	import matplotlib.pyplot as plt
	weight = wei.getA() # 将矩阵转换为 ndarray
	dataset,labelset = loaddataset() #list
	n = shape(dataset)[0] # 返回行数 即样本个数
	xcord1 = [];ycord1 = []
	xcord2 = [];ycord2 = [] #初始化两类数据的坐标
	for i in range(n): #按照分类将两类数据点分开 方便画图
		if labelset[i] == 1:
			xcord1.append(dataset[i][1]);ycord1.append(dataset[i][2])
		else:
			xcord2.append(dataset[i][1]);ycord2.append(dataset[i][2])
	fig = plt.figure()
	ax = fig.add_subplot(111) #
	#ax.scatter(xcord,ycord, c=colors, s=markers)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=20,c='green')
	x = arange(-3.0,3.0,0.1) 
	y = (-weight[0]-weight[1]*x)/weight[2] #边界是sigmoid(z)=0.5,即z=0，即W.T*X=0，则根据特征1x的扫描，计算特征2y的值
	ax.plot(x,y)
	plt.xlabel('X1');plt.ylabel('X2')
	plt.show()

# 梯度上升法的计算代价太大：每次更新恢复系数时都需要遍历整个数据集（即每次每个回归系数的更新均需要所有训练样本中相应的特征）
# 改进的一种方法：一次仅用一个样本点来更新回归系数——》随机梯度上升法
# 随机梯度上升法可以在新样本到来时对分类器进行增量式更新，so 是在线学习算法【即可以实现随着数据的到来而依次处理】
# 在线学习对应：批处理【一次处理所有数据】
'''
伪代码：
所有回归系数初始化为1
对数据集中每个样本：
	计算该样本的梯度
	使用α×gradient更新回归系数值【注意，梯度上升法是一次更新整个回归系数矢量，而此处是一个值】
返回回归系数值
'''
def randgradascent(dataset,labelset):
	m,n=shape(dataset)
	ardataset = array(dataset)
	alpha=0.01
	weights = ones(n) # 一行n列的一个一维array
	for i in range(m):
		h = sigmoid(sum(ardataset[i]*weights)) # 此处是点乘 注意得到的只是一个结果
		error = labelset[i]-h
		weights = weights + alpha*error*ardataset[i] #每次更新仅用一个样本，则更新次数为训练样本个数
	return mat(weights).T  # 注意输出的weight需要与之前梯度上升法的输出 一致，这样其他函数的引用才不会出问题
		
# import sz_c5
# dataset,labelset = sz_c5.loaddataset()
# w = sz_c5.randgradascent(dataset,labelset)
# sz_c5.plotbestfit(w)

# 上述随机法可以通过对训练集多次迭代来实现w参数的稳定，但是即使大的波动停止后，仍会有小的周期波动
# 改进1：alpha在每次迭代都会调整，这可以缓解数据波动或者是高频波动；但是若是有一个新的样本到来，则可以适当增大对应的alpha值，以增大改数据的影响
#        可以使得参数收敛速度增大！
# 改进2：随机选取样本来更新回归系数，可以减少周期性波动
def randgradascent1(dataset,labelset,numIter=150): # 输入的参数增加一项迭代数目，默认150
	m,n = shape(dataset)
	dataset = array(dataset) # 从list转换成array
	weights = ones(n)
	for j in range(numIter): #每次迭代
		dataindex = list(range(m)) # range 不支持del 某项，需要将其转换成list
		for i in range(m): #每个训练样本
			alpha = 4/(1.0+j+i)+0.01 # 每次迭代时需要调整alpha值，随着迭代次数的增大而减小!!!
			randindex = int(random.uniform(0,len(dataindex))) # 随机生成下标
			h = sigmoid(sum(dataset[randindex]*weights))
			error = labelset[randindex]-h
			weights = weights+alpha*error*dataset[randindex]
			del(dataindex[randindex])
	return mat(weights).T
# reload(sz_c5)
# dataset,labelset = sz_c5.loaddataset()
# w = sz_c5.randgradascent1(dataset,labelset)
# sz_c5.plotbestfit(w)			
'''
示例：从疝气病症预测病马的死亡率
'''
# 数据的预处理！！
# 1）缺失值的处理：numpy数据类型不允许含缺失值；
#		logistic回归中可以用实数0来替换缺失值，这样根据weight更新公式可以看出，缺失值不会对其造成影响
# 		则可以保留现有数据，同时不需要优化算法，且数据集中的特征值一般不会为0；
# 2）若一条数据的类别标签缺失：对于logistic回归来说，可以简单地将这条数据丢弃；但是对于KNN可能就不太可行？？？

# 数据预处理（自写） 从网上下载的原始数据似乎和提供的数据不太一样，多了几列不相关的特征，应该是需要自己去掉的
from numpy import *
def preproc():
	filename = r'.\data\horse-colic.data'
	fr = open(filename)
	origdata = fr.readlines()
	fid = open('1.txt','w')
	sep = ','
	for line in origdata:
		nline=[]
		items = line.strip().split(' ')
		if items[0] != '?': # 每行第一项为label，是？时要把这条数据删掉
			for item in items:
				if item != '?': nline.append(item)
				else: nline.append('0') # 若特征数据缺失，则用0替代
			sline = sep.join(nline) # 把list转换成以sep ','相隔的字符串
			fid.write(sline) # 一次写入一行
			fid.write('\n') # 换行
	fr.close()
	fid.close()

# 测试算法：
# 思路：把测试集中每个特征向量乘最优化方法得到的回归参数，求和，代入到sigmoid函数中，结果大于0.5分类为1，否则为0
def classify(inx,weight): # 输入的不是矩阵 而是list或者array；输出分类 1.0 or 0.0
	# prob = sigmoid(sum(inx*weight)) # 此处不是点乘 list或者array与matrix相乘是按照matrix的方式来的，得到的结果是matrix！！
	prob = sigmoid(inx*weight) 
	if prob>0.5:return 1.0
	else: return 0.0
	
def colictest():
	# 读入数据：训练集和测试集
	frtrain = open(r'horseColicTraining.txt')
	frtest = open(r'horseColicTest.txt')
	trainset = [];trainlabel = []
	for line in frtrain.readlines():
		items = line.strip().split('\t')
		aline = []
		for i in range(21): # 0到20
			aline.append(float(items[i])) #将字符串转换为浮点数
		trainset.append(aline)
		trainlabel.append(float(items[21])) # 最后一列为label，前面为feature
	# 训练回归参数
	weight = randgradascent1(trainset,trainlabel,500) #随机梯度上升法
	# weight = gradascent(trainset,trainlabel) # 梯度上升法
	'''
	！！！考虑一个问题：梯度上升法得到的结果是否一定会比随机的好？？？ 0.283582
	不一定吧，每次用所有训练集进行更新，也很可能造成过度匹配的问题？
	'''
	# 开始测试
	errorcount = 0; numtest = 0.0
	for line in frtest.readlines():
		numtest += 1.0
		items = line.strip().split('\t')
		inx = []
		for i in range(21):
			inx.append(float(items[i]))
		if int(classify(inx,weight)) != int(items[21]):
			errorcount += 1
	errorrate = errorcount/numtest
	print('the error rate of this test is: %f' % (errorrate))
	return errorrate
# 随机梯度上升法，则每次运行训练得到的结果和测试结果会有不同，多次测试求均值
def multitest():
	num = 10;errorsum=0.0
	for k in range(num):
		errorsum += colictest()
	print('after %d iterations the average error rate is: %f' % (num, errorsum/num))
# 通过调整训练中的迭代次数和步长，平均错误率可以降低，—— 第7章！！	
	
