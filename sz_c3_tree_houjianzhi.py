# 机器学习实战 学习记录
# Chapter 3 决策树 基尼系数 + 后剪枝
# coding='UTF-8'
from numpy import * 
import operator
import pandas as pd
from math import log
import re

def calgini(dataset):
	num = len(dataset)
	labelcounts = {}
	for line in dataset: # 统计数据集中每个类别出现的概率
		currentlabel = line[-1]
		labelcounts[currentlabel] = labelcounts.get(currentlabel,0)+1
	Gini = 1.0
	for key in labelcounts: # 利用上述统计的概率计算Gini系数
		prob = float(labelcounts[key]/num)
		Gini -= prob*prob
	return Gini

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
def splitContinuousDataSet(dataset,axis,value,direction):  
	retDataSet=[]  
	for featVec in dataset:  
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
	baseshang = calgini(dataset)
	bestgini = 10000.0 #基尼系数是越小越好
	bestfeature = -1
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
				newEntropy+=prob0*calgini(subDataSet0)  
				prob1=len(subDataSet1)/float(len(dataset))  
				newEntropy+=prob1*calgini(subDataSet1)  
				if newEntropy<bestSplitEntropy:  
					bestSplitEntropy=newEntropy  
					bestSplit=j  
			#用字典记录当前特征的最佳划分点  
			bestSplitDict[labels[i]]=splitList[bestSplit]  
			gini = newEntropy
		else: #离散
			valuelist = set(featlist) # 提取出列中所有值类别
			newgini = 0.0
			for value in valuelist:
				newdataset = splitdataset(dataset,i,value)
				prob = len(newdataset)/len(dataset) #每种value出现的频率
				gain = calgini(newdataset)
				newgini += prob*gain
				# print('%d, %f, %f, %f, %f,' % (i,value,prob,gain,newshang)) #检查程序问题
			gini = newgini
			# print('shanggain %f' % (shanggain))
		if gini < bestgini:
			bestgini = gini
			bestfeature = i
	# 判断所选属性的值是否是连续的，若是，需要将属性值进行二值化处理
	#即是否小于等于bestSplitValue  
	if type(dataset[0][bestfeature]).__name__=='float' or type(dataset[0][bestfeature]).__name__=='int':        
		bestSplitValue=bestSplitDict[labels[bestfeature]]          
		labels[bestfeature]=labels[bestfeature]+'<='+str(bestSplitValue)  
		for i in range(shape(dataset)[0]):  
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
	
def createtree(dataset,labels,data_full,labels_full,data_test): # 输入的labels仅是前面feature，没有最后的分类结果label
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
		mytree[bestfeaturelabel][value] = createtree(newdataset,sublabels,data_full,labels_full,data_test) # 递归
	# 有可能子集中没有包括所有可能属性值，这会使得以后可能有些样本无法分类
	# 所以，需要记录特征的所有属性，为子集没有的属性分配个类别
	if type(dataset[0][bestfeature]).__name__=='str':  
		for value in uniqueValsFull: # 没有出现在子集中的属性  
			mytree[bestfeaturelabel][value] = majority(classlist)
	return mytree
# dataset,labels = sz_c3_tree.createdataset()
# mytree = sz_c3_tree.createtree(dataset,labels)
'''
使用决策树构建分类器，实际应用中存储分类器
利用训练数据构造决策树，然后将其用于实际数据的分类
'''
def classify(inputtree,featlabels,testvec): # featlabels是与实际数据对应的列名（特征名）
	firstfeat = list(inputtree.keys())[0]
	
	if '<=' in firstfeat:
		featvalue=float(re.compile("(<=.+)").search(firstfeat).group()[2:])
		featkey=re.compile("(.+<=)").search(firstfeat).group()[:-2]
		secdict=inputtree[firstfeat]
		featindex = featlabels.index(featkey)
		if testvec[featindex] <= featvalue:
			judge = 1
		else:
			judge = 0
		for value in secdict.keys():
			if judge==int(key):
				subtree = secdict[value]
				if type(subtree).__name__ == 'dict':
					classlabel = classify(subtree,featlabels,testvec)
				else:
					classlabel = subtree
	else:
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
	
def testing(myTree,data_test,labels):  
	error=0.0  
	for i in range(len(data_test)):  
		if classify(myTree,labels,data_test[i])!=data_test[i][-1]:  
			error+=1  
	print('myTree %d' % (error))  
	return float(error)  
      
def testingMajor(major,data_test):  
	error=0.0  
	for i in range(len(data_test)):  
		if major!=data_test[i][-1]:  
			error+=1  
	print('major %d' % (error))  
	return float(error)
	
#后剪枝  
def postPruningTree(inputTree,dataset,data_test,labels):  
	firstStr=list(inputTree.keys())[0]  
	secondDict=inputTree[firstStr]  
	classList=[example[-1] for example in dataset]  
	featkey= firstStr  
	if '<=' in firstStr:  
		featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]  
		featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])  
	labelIndex=labels.index(featkey)  
	temp_labels=labels[:]  
	del(labels[labelIndex])  
	for key in secondDict.keys():  
		if type(secondDict[key]).__name__=='dict':  
			if type(dataset[0][labelIndex]).__name__=='str':  
				inputTree[firstStr][key]=postPruningTree(secondDict[key],\
				splitdataset(dataset,labelIndex,key),splitdataset(data_test,labelIndex,key),labels)  
			else:  
				inputTree[firstStr][key]=postPruningTree(secondDict[key],\
				splitContinuousDataSet(dataset,labelIndex,featvalue,key),\
				splitContinuousDataSet(data_test,labelIndex,featvalue,key),\
				labels)  
	if testing(inputTree,data_test,temp_labels)<=testingMajor(majority(classList),data_test):  
		return inputTree  
	return majority(classList) 
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
filename = 'lenses.txt'
fid = open(filename)
data = fid.readlines()
dataset = []
for line in data[:-10]:
	line = line.strip()
	line1 = line.split('\t')[:]
	dataset.append(line1)
data_test = []
for line in data[-10:-1]:
	line = line.strip()
	line1 = line.split('\t')[:]
	data_test.append(line1)
# dataset = [line.strip().split('\t') for line in fid.readlines()] #跟上面一样的效果，不过简洁很多
lenseslabels = ['age','prescript','astigmatic','tearrate']
data_full = dataset[:]
labels_full = lenseslabels[:]
lensestree = sz_c3_tree_houjianzhi.createtree(dataset,lenseslabels,data_full,labels_full,data_test)
mytree = sz_c3_tree_houjianzhi.postPruningTree(lensestree,dataset,data_test,labels_full)
sz_c3_treeplot.createplot(mytree)
'''
