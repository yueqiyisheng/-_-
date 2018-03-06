# 机器学习实战 学习记录
# Chapter 3 决策树 本文件的方法基于ID3算法，其他构造算法有C4.5、CART等
# coding='UTF-8'
'''
优点：计算复杂度不高，输出结果易于理解，对中间值缺失不敏感，可以处理不相关特征数据
缺点：可能产生过度匹配问题【可以通过裁剪决策树，去掉一些产生信息少的叶节点，将其并入其他叶节点中————> Chapter 9】
适用于：数值（需要离散化） 和 标称
'''

'''
创建分支的伪代码 createBranch() 递归函数

检测数据集中的每个子项是否属于同一分类：
	if so return 类标签；
	else 
		寻找划分数据集的最好特征
		划分数据集
		创建分支节点
			for 每个划分的子集
				调用 createBranch并增加返回结果到分支节点中
		return 分支节点
'''
'''
计算给定数据集的信息熵
'''
from numpy import *
import operator
from math import log
def calxiangnongshang(dataset):
	num = len(dataset)
	labelcounts = {}
	for line in dataset: # 统计数据集中每个类别出现的概率
		currentlabel = line[-1]
		labelcounts[currentlabel] = labelcounts.get(currentlabel,0)+1
	xiangnongshang = 0.0
	for key in labelcounts: # 利用上述统计的概率计算香农熵
		prob = float(labelcounts[key]/num)
		xiangnongshang -= prob*log(prob,2)
	return xiangnongshang

# 简单的示例：鱼鉴定
# 生成数据集
def createdataset():
	dataset = [[1,1,'y'],[1,1,'y'],[1,0,'n'],[0,1,'n'],[0,1,'n']]
	labels = ['no surfacing','flippers']
	return dataset,labels

'''
按照给定特征划分数据集
'''
# 按照某列axis的值value把数据提取出来
def splitdataset(dataset,axis,value): 
	newdataset = []
	for line in dataset:
		if line[axis] == value:
			reducedline = line[:axis]
			reducedline.extend(line[axis+1:])
			newdataset.append(reducedline)
	return newdataset
	
#对连续变量划分数据集，direction规定划分的方向，  
#决定是划分出小于value的数据样本还是大于value的数据样本集  
def splitContinuousDataSet(dataSet,axis,value,direction):  
	retDataSet=[]  
	for featVec in dataSet:  
		if direction==0:  
			if featVec[axis]>value:  
				reducedFeatVec=featVec[:axis]  
				reducedFeatVec.extend(featVec[axis+1:])  
				retDataSet.append(reducedFeatVec)  
		else:  
			if featVec[axis]<=value:  
				reducedFeatVec=featVec[:axis]  
				reducedFeatVec.extend(featVec[axis+1:])  
				retDataSet.append(reducedFeatVec)  
	return retDataSet 

# sz_c3_tree.splitdataset(dataset,0,0)
'''
遍历数据集，计算香农熵，选择最好的数据集划分方式
'''
def choosebestfeaturetosplit(dataset,labels): #返回列号
	numfeature = len(dataset[0])-1
	baseshang = calxiangnongshang(dataset)
	bestshanggain = 0.0
	bestfeature = -1
	bestSplitDict = {}
	for i in range(numfeature):
		featlist = [line[i] for line in dataset] # 提取出axis那列
		# 区分连续还是离散
		# 如果连续
		if type(featlist[0]).__name__=='float' or type(featlist[0]).__name__=='int':
			#产生n-1个候选划分点  
			sortfeatList=sorted(featlist)  
			splitList=[]  
			for j in range(len(sortfeatList)-1):  
				splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)
				
			bestSplitEntropy=10000  
			slen=len(splitList)  
			#求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点  
			for j in range(slen):  
				value=splitList[j]  
				newEntropy=0.0  
				subDataSet0=splitContinuousDataSet(dataset,i,value,0)  
				subDataSet1=splitContinuousDataSet(dataset,i,value,1)  
				prob0=len(subDataSet0)/float(len(dataset))  
				newEntropy+=prob0*calxiangnongshang(subDataSet0)  
				prob1=len(subDataSet1)/float(len(dataset))  
				newEntropy+=prob1*calxiangnongshang(subDataSet1)  
				if newEntropy<bestSplitEntropy:  
					bestSplitEntropy=newEntropy  
					bestSplit=j  
			#用字典记录当前特征的最佳划分点  
			bestSplitDict[labels[i]]=splitList[bestSplit]  
			shanggain=baseshang-bestSplitEntropy
		else: #离散
			valuelist = set(featlist) # 提取出列中所有值类别
			newshang = 0.0
			for value in valuelist:
				newdataset = splitdataset(dataset,i,value)
				prob = len(newdataset)/len(dataset) #每种value出现的频率
				gain = calxiangnongshang(newdataset)
				newshang += prob*gain
				# print('%d, %f, %f, %f, %f,' % (i,value,prob,gain,newshang)) #检查程序问题
			shanggain = baseshang-newshang
			# print('shanggain %f' % (shanggain))
		if shanggain>bestshanggain:
			bestshanggain = shanggain
			bestfeature = i
	# 判断所选属性的值是否是连续的，若是，需要将属性值进行二值化处理
	#即是否小于等于bestSplitValue  
	if type(dataset[0][bestfeature]).__name__=='float' or type(dataset[0][bestfeature]).__name__=='int':        
		bestSplitValue=bestSplitDict[labels[bestfeature]]          
		labels[bestfeature]=labels[bestfeature]+'<='+str(bestSplitValue)  
		for i in range((size(dataset,0))):  
			if dataset[i][bestfeature]<=bestSplitValue:  
				dataset[i][bestfeature]=1  
			else:  
				dataset[i][bestfeature]=0 
	return bestfeature

#dataset,labels = sz_c3_tree.createdataset()
#sz_c3_tree.choosebestfeaturetosplit(dataset)
'''
递归构建决策树
'''
def majority(classlist): #得到类别列中最多的类
	classcount = {}
	for vote in classlist:
		classcount[vote] = classcount.get(vote,0) + 1
	sortedclasscount = sorted(classcount,key = operator.itemgetter(1),reverse = True) #sorted函数的使用，operator.itemgetter的使用
	return sortedclasscount[0][0]
	
def createtree(dataset,labels,data_full,labels_full): # 输入的labels仅是前面feature，没有最后的分类结果label
	classlist = [line[-1] for line in dataset]
	if classlist.count(classlist[0]) == len(classlist): #类别完全相同，停止划分
		return classlist[0]
	if len(dataset[0]) == 1: # 遍历完所有特征后，返回类别中最多的类别 ‘表决器’
		return majority(classlist)
	# 选择最好的数据集划分属性，若是连续值的，则将其二值化
	bestfeature = choosebestfeaturetosplit(dataset,labels) #返回的列号 
	bestfeaturelabel = labels[bestfeature]
	mytree = {bestfeaturelabel:{}}
	# 
	featValues=[example[bestfeature] for example in dataset]  
	uniqueVals=set(featValues)  
	if type(dataset[0][bestfeature]).__name__=='str': # 对于离散值的属性，进行下述操作  
		currentlabel=labels_full.index(labels[bestfeature])  
		featValuesFull=[example[currentlabel] for example in data_full]  
		uniqueValsFull=set(featValuesFull)  
	labelscopy = labels[:]  #若直接将labels进行下面的del，则会在labels上操作，把labels本身改变，会影响函数外labels的使用
	del(labelscopy[bestfeature]) #删除已分配的特征
	#针对bestfeature的每个取值，划分出一个子树。
	for value in uniqueVals:
		sublabels = labelscopy[:]
		if type(dataset[0][bestfeature]).__name__=='str':  
			uniqueValsFull.remove(value)
		newdataset = splitdataset(dataset,bestfeature,value)
		mytree[bestfeaturelabel][value] = createtree(newdataset,sublabels,data_full,labels_full ) # 递归
	# 有可能子集中没有包括所有可能属性值，这会使得以后可能有些样本无法分类
	# 所以，需要记录特征的所有属性，为子集没有的属性分配个类别
	if type(dataset[0][bestfeature]).__name__=='str':  
		for value in uniqueValsFull: # 没有出现在子集中的属性  
			myTree[bestfeaturelabel][value] = majority(classlist)  
	return mytree
# dataset,labels = sz_c3_tree.createdataset()
# mytree = sz_c3_tree.createtree(dataset,labels)
'''
使用决策树构建分类器，实际应用中存储分类器
利用训练数据构造决策树，然后将其用于实际数据的分类
'''
def classify(inputtree,featlabels,testvec): # featlabels是与实际数据对应的列名（特征名）
	firstfeat = list(inputtree.keys())[0]
	secdict = inputtree[firstfeat]
	featindex = featlabels.index(firstfeat)
	for value in secdict.keys():
		if testvec[featindex] == value:
			subtree = secdict[value]
			if type(subtree).__name__ == 'dict':
				classlabel = classify(subtree,featlabels,testvec)
			else:
				classlabel = subtree
	return classlabel
# dataset,labels = sz_c3_tree.createdataset()
# mytree = sz_c3_tree.createtree(dataset,labels)
# sz_c3_tree.classify(mytree,labels,[1,0])
'''
每次构造决策树是很耗时的，所以需要在磁盘上将构造好的决策树进行存储
需要利用到pickle模块
对象序列化：它是一个将任意复杂的对象转成对象的文本或二进制表示的过程
'''
def storetree(inputtree,filename):
	import pickle
	fw = open(filename,'wb') # 在python3以上版本中，如果要用存储器，那么读写文件都要用‘rb’和'wb'模式!!
	pickle.dump(inputtree,fw)  #pickle.dumps(obj)将对象序列化为字符串
	fw.close()

def grabtree(filename):
	import pickle
	fr = open(filename,'rb')
	tree = pickle.load(fr) #pickle.loads(str)将字符串返回成对象
	return tree
# filename = 'fishclasstree.txt'
# sz_c3_tree.storetree(mytree,filename)
# mytree = sz_c3_tree.grabtree(filename)
'''
示例：使用决策树预测隐形眼镜类型
'''
'''
filename = 'wine.data'
fid = open(filename)
data = fid.readlines()
dataset0 = []
for line in data:
	line = line.strip()
	line1 = line.split(',')[:]
	dataset0.append(line1)
for i in range((size(dataset0,0))):
	for j in range((size(dataset0,1)-1)):
		dataset[i][j] = float(dataset0[i][j+1])
	dataset[i][size(dataset0,1)-1] = float(dataset0[i][0])

lenseslabels = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',\
'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315','Proline']
data_full = dataset[:]
labels_full = lenseslabels[:]
lensestree = sz_c3_tree_addlianxu.createtree(dataset,lenseslabels,data_full,labels_full)
sz_c3_treeplot.createplot(lensestree)
'''
