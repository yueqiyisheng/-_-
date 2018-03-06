# 机器学习实战 学习记录
# Chapter 7 利用AdaBoost元算法提高分类性能 
# coding='UTF-8'
from numpy import *
def loadsimdata():
	dataset = [[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]]
	classlabel = [1.0,1.0,-1.0,-1.0,1.0]
	return dataset,classlabel
# 单层决策树
# 输入：数据集，特征列号，阈值，分类方式
def stumpclassify(dataset,dimen,threshval,threshineq):
	datamat = mat(dataset)
	retarray = ones((shape(datamat)[0],1)) # 初始化分类结果
	if threshineq == 'it':
		retarray[datamat[:,dimen] <= threshval] = -1.0 # 数据集中指定特征的数据不大于阈值，分类为-1
	else:
		retarray[datamat[:,dimen] > threshval] = -1.0
	return retarray
# 寻找最佳单层决策树
# D是数据的权重向量
def buildstump(dataset,classlabel,D):
	datamat = mat(dataset)
	labelmat = mat(classlabel).T
	m,n = shape(datamat)
	numstep = 10.0
	beststump = {};bestclassest = mat(zeros((m,1)))
	minerror = inf
	for i in range(n): # 对每个特征
		rangemin = datamat[:,i].min();rangemax = datamat[:,i].max() #特征数据的范围
		stepsize = (rangemax - rangemin)/numstep # 步长
		for j in range(-1,int(numstep)+1):
			for inequal in ['it','gt']: # 分类方式
				threshval = rangemin + stepsize*float(j) # 步增的阈值
				predictvals = stumpclassify(datamat,i,threshval,inequal)
				err = mat(ones((m,1)))
				err[mat(predictvals) == labelmat] = 0 #注意里面比较的两个必须是矩阵，而stumpclassify()输出的是array，需要转换成mat，否则会报错
				#for k in range(m):
				#	if predictvals[k] == labelmat[k]:
				#		err[k] = 0
				weighte = D.T*err #计算加权错误率
				# print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error: %.3f' % (i,threshval,inequal,weighte))
				if weighte < minerror:
					minerror = weighte
					bestclassest = predictvals.copy()
					beststump['dim'] = i
					beststump['thresh'] = threshval
					beststump['ineq'] = inequal
	return beststump,minerror,bestclassest
'''
datamat,classlabel = sz_c7.loadsimdata()
D = mat(ones((5,1))/5)
sz_c7.buildstump(datamat,classlabel,D)
'''
# 完整AdaBoost算法
'''
对每次迭代：
	利用buildstump（）找到最优单层决策树
	将其加入到单层决策树数组中
	计算alpha
	计算新的D
	更新累计类别估计值  ？？
	若错误率为0，则退出循环
'''
def adaboosttrain(dataset,classlabel,numit = 40):
	datamat = mat(dataset)
	weakclass = []
	m = list(shape(datamat))[0]
	D = mat(ones((m,1))/m) #初始化加权矩阵
	aggclassest = mat(zeros((m,1)))
	for i in range(numit): #每次迭代
		beststump,error,classest = buildstump(datamat,classlabel,D) #单层决策树
		# print('D:',D.T)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16))) # 计算alpha，max用于防止除零溢出
		beststump['alpha'] = alpha
		weakclass.append(beststump) #将决策树加入到弱分类器组中
		# print('classest:',classest.T)
		expon = multiply(-1*alpha*mat(classlabel).T,classest)
		D = multiply(D,exp(expon)) 
		D = D/D.sum() # 更新加权向量D
		aggclassest += alpha*classest #累计分类估计
		# print('aggclassest:',aggclassest.T)
		aggerror = multiply(sign(aggclassest) != mat(classlabel).T, ones((m,1))) #累计误差
		errorrate = aggerror.sum()/m
		print('total error: ',errorrate)
		if errorrate == 0.0:
			break
	return weakclass,aggclassest #以aggclassest的符号来判断分类，则绝对值越大，表明成功分类的可能性更大

'''
c = sz_c7.adaboosttrain(datamat,classlabel,9)
'''
# 基于AdaBoost的分类
# 测试:输入待分类的数据 和 已经训练好的分类器
def adaclass(datatoclass,classifier):
	datamat = mat(datatoclass)
	m = list(shape(datamat))[0]
	aggclassest = mat(zeros((m,1)))
	for i in range(len(classifier)): # 单层决策树的数量
		classest = stumpclassify(datamat,classifier[i]['dim'],classifier[i]['thresh'],classifier[i]['ineq'])
		aggclassest += classifier[i]['alpha']*classest #累计分类估计
		# print(aggclassest)
	return sign(aggclassest)
'''
reload(sz_c7)
datamat,classlabel = sz_c7.loadsimdata()
classifier = sz_c7.adaboosttrain(datamat,classlabel,30)
sz_c7.adaclass([0,0],classifier)
sz_c7.adaclass([[5,5],[0,0]],classifier)
'''
# 将AdaBoost应用到一个难数据集上
# 第5章中使用logistic回归预测患有疝病的马是否能存活
# 此处将AdaBoost应用于此数据集
# 注意：logist回归中类别标签是0,1；此处需要改成-1和+1（新数据已经改了）

# 常见的读入数据 !!!
def loaddata(filename):
	numfeat = len(open(filename).readline().strip().split('\t'))
	dataset = [];labelset = []
	fr = open(filename)
	for line in fr.readlines():
		linearr = []
		temp = line.strip().split('\t')
		for i in range(numfeat-1):
			linearr.append(float(temp[i]))	
		dataset.append(linearr)
		labelset.append(float(temp[-1])) #最后一列是label
	return dataset,labelset

'''
filename = r'.\horseColicTraining2.txt'
datamat,labelmat = sz_c7.loaddata(filename)
classifier = sz_c7.adaboosttrain(datamat,labelmat,10)

# 测试
filename = r'.\horseColicTest2.txt'
testdata,testlabel = sz_c7.loaddata(filename)
predict = sz_c7.adaclass(testdata,classifier)
error = mat(ones((len(testdata),1)))
nerr = error[predict != mat(testlabel).T].sum()
errorrate = nerr/len(testdata)
'''
# 关于非均衡分类问题的介绍与处理方法的简介

# ROC曲线
# 横轴：实际为错误的样本中判断为正确的所占比例
# 纵轴：实际为正确的样本中判断为正确的所占比例
def plotroc(predstrength,classlabel):
	import matplotlib.pyplot as plt
	cur = (1.0,1.0) #绘制光标的位置 初始位置：将全部样本判断为正确
	ysum = 0.0 # 用于计算AUC，即ROC曲线面积
	numpos = sum(mat(classlabel) == 1.0) #实际为+1的数量
	ystep = 1/float(numpos)
	xstep = 1/float(len(classlabel) - numpos)
	sortedindicies = predstrength.argsort() #获取排好序的索引，返回的是array
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedindicies.tolist()[0]: #tolist返回的是一个嵌套，
		if classlabel[index] == 1.0:
			delx = 0; dely = ystep
		else:
			delx = xstep; dely = 0
			ysum += cur[1]
		ax.plot([cur[0],cur[0]-delx],[cur[1],cur[1]-dely],c='b')
		cur = (cur[0]-delx,cur[1]-dely)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	ax.axis([0,1,0,1])
	plt.show()
	print('the area under the curve is: %f' % (ysum*xstep))
'''
reload(sz_c7)
filename = r'.\horseColicTraining2.txt'
datamat,labelmat = sz_c7.loaddata(filename)
classifier, aggclassest = sz_c7.adaboosttrain(datamat,labelmat,10)
sz_c7.plotroc(aggclassest.T,labelmat)
'''

