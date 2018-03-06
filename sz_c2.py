# 机器学习实战 学习记录
# Chapter 2 K-近邻算法（KNN） 
# coding='UTF-8'
'''
KNN伪代码：
（1）计算当前点与已知数据类别的数据集中的点的距离；
（2）按照距离递增排序；
（3）取前k个点（k人工指定，需要考虑设置为多少）；
（4）确定前k个点中所在类别出现的频率；
（5）选择出现频率最高的类别作为当前点的预测分类。
'''
'''
优点：精度高，对异常数据不敏感，无数据输入假定；
缺点：计算复杂度高、空间复杂度高
适用范围：数值型 和 标称型（即有限类别）
'''
from imp import reload #修改。py文件后需要reload一下
from numpy import *
import operator
from collections import Counter
from os import listdir
'''
简单的例子
'''
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,Dataset,labels,k):
	dataset_size = Dataset.shape[0] #返回数据集的行数
	diff = tile(inX,(dataset_size,1))-Dataset #分量差值
	sqdiff = diff**2
	sumsq = sqdiff.sum(axis=1) #将这一列求和
	distance = sumsq**0.5
	sortdis = distance.argsort() #得到从小到大排列后的序号 
	classcount = {}
	for i in range(k):
		lab = labels[sortdis[i]]
		classcount[lab] = classcount.get(lab,0)+1 #get若有lab的key，则返回对应的值，若没有，则返回后面的0
	sortclass = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
	# 注意不能直接classcount.values()，这样返回的就只有值，而没有想要的key了，所以需要用key来指定排序的方式
	return sortclass[0][0]
	
'''
示例1：使用KNN改进约会网站的配对效果
'''
# 从文本文件中读入数据，处理数据格式
# 输出为：训练样本矩阵 和 类标签
# 文本文件读入的数据是字符串格式的，包含\t,\n等
# 需要先了解下文件中读入后的格式

def file2matrix(filename):
	fr = open(filename)
	arraylines = fr.readlines() #字符串格式
	numberoflines = len(arraylines)
	dataset = zeros((numberoflines,3)) #初始化数据集矩阵
	label = []
	index = 0
	for line in arraylines:
		line = line.strip() # 移除字符串头尾指定的字符，不指定字符串时可将末尾的回车\n移除
		listfromline = line.split('\t') # 以'\t'将字符串分割成几部分
		dataset[index,:] = listfromline[0:3]
		label.append(listfromline[-1])
		index += 1
	items = Counter(label) #确定类别 字典
	items_list = list(items) #将字典的键转换成list
	labels = list(map(lambda x:items_list.index(x)+1,label)) #将字符串的label转换成int
	return dataset,labels,items_list

'''
# 制作原始数据的散点图（可视化）
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111) #子图 1行1列的第1个，(2,3,1)等也可
# ax.scatter(dataset[:,1],dataset[:,2]) #第2列和第3列
# ax.scatter(dataset[:,0],dataset[:,1],15.0*array(labels),15.0*array(labels))
l1 =[i for i,a in enumerate(labels) if a==1]
ax.scatter(dataset[l1,0],dataset[l1,1],color='r',s=15.0,label = 'a')
l2 =[i for i,a in enumerate(labels) if a==2]
ax.scatter(dataset[l2,0],dataset[l2,1],color='m',s=30.0,label = 'b')
l3 =[i for i,a in enumerate(labels) if a==3]
ax.scatter(dataset[l3,0],dataset[l3,1],color='c',s=45.0,label = 'c')
#ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
ax.legend()
plt.show()		
'''
#数据归一化 到[0,1]或者[-1,1]
def autoNorm(dataset):
	dmin = dataset.min(0) #按列
	dmax = dataset.max(0) 
	ran = dmax-dmin
	normdataset = zeros(shape(dataset))
	m = dataset.shape[0]
	normdataset = dataset - tile(dmin,(m,1))
	normdataset = normdataset/tile(ran,(m,1))
	return normdataset,ran,dmin

# 分类器针对约会网站的测试代码
def datingClassTest():
	dataset,labels,items_list = file2matrix('datingTestSet.txt')
	ratio = 0.2 #取xx%作为测试数据集
	normmat,ran,dmin = autoNorm(dataset)
	m = dataset.shape[0]
	mtest = int(m*ratio)
	errorcount = 0.0
	for i in range(mtest):
		classifierresult = classify0(normmat[i,:],normmat[mtest:m,:],labels[mtest:m],3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierresult,labels[i]))
		if classifierresult != labels[i]:
			errorcount += 1.0
	print("the total error rate is: %f" % (errorcount/m)) #里面这个小括号是必须的，否则会报错

# datingClassTest()	

# 使用算法：构建完整可用系统
def classifyperson():
	# resultlist = ['not at all','in small doses','in large doses']
	percent = float(input(\
	"percentage of time spent playing video games?"))
	ffmiles = float(input("frequent flier miles earned per year?"))
	icecream = float(input("liters of ice cream consumed per year?"))
	dataset,labels,items_list = file2matrix('datingTestSet.txt')
	normmat,ran,dmin = autoNorm(dataset)
	inx = [ffmiles,percent,icecream] #注意参数顺序
	classifierresult = classify0((inx-dmin)/ran,normmat,labels,3)
	print("You will probably like this person: ",items_list[classifierresult-1])
# classifyperson()
	
'''
示例2：手写识别系统（数字0-9）
已经将需要识别的数字转换成32*32像素的黑白图像，存储在文本文件中
'''
# classify0的输入数据点是一个向量，需要将图像的矩阵数据（32*32）转换成一个向量1024
def img2vector(filename):
	vec = zeros((1,1024))
	fr = open(filename)
	#依次读取前32行数据，并将每行的前32个字符存在向量中
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			vec[0,32*i+j] = int(line[j]) #不能将一行32个一起int，然后传过去
	return vec
	
def handwritingClasstest():
	hwlabel = []
	trainfilelist = listdir(r'trainingDigits')
	# 读取文件夹中的目录内容 注意 r 
	m = len(trainfilelist)
	dataset = zeros((m,1024))
	# 生成训练数据集
	for i in range(m):
		# 从文件名中解析中分类数字
		filenamestr = trainfilelist[i]
		classnum = int(filenamestr[0])
		hwlabel.append(classnum)
		# 读取每个文件中的数据
		dataset[i,:] = img2vector(r'.\trainingDigits\%s' % filenamestr)
	
	# 读取测试数据集
	testfilelist = listdir(r'.\testDigits')
	errorcount = 0.0
	n = len(testfilelist)
	for i in range(n):
		realclassnum = int(testfilelist[i][0])
		inx = img2vector(r'.\testDigits\%s' % testfilelist[i])
		classnumresult = classify0(inx,dataset,hwlabel,3)
		print('the classifier came back with: %d, the real answer is: %d' % (classnumresult,realclassnum))
		if classnumresult != realclassnum:
			errorcount += 1.0
	print('the total number of errors is: %d' % (errorcount))
	print('the total error rate is: %f' % (errorcount/n))

	
	
	
	
		
	
