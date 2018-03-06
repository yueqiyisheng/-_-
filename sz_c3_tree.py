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
# sz_c3_tree.splitdataset(dataset,0,0)
'''
遍历数据集，计算香农熵，选择最好的数据集划分方式
'''
def choosebestfeaturetosplit(dataset): #返回列号
	numfeature = len(dataset[0])-1
	baseshang = calxiangnongshang(dataset)
	bestshanggain = 0.0
	bestfeature = -1
	for i in range(numfeature):
		featlist = [line[i] for line in dataset] # 提取出axis那列
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
	
def createtree(dataset,labels): # 输入的labels仅是前面feature，没有最后的分类结果label
	classlist = [line[-1] for line in dataset]
	if classlist.count(classlist[0]) == len(classlist): #类别完全相同，停止划分
		return classlist[0]
	if len(dataset[0]) == 1: # 遍历完所有特征后，返回类别中最多的类别 ‘表决器’
		return majority(classlist)
	bestfeature = choosebestfeaturetosplit(dataset) #返回的列号
	bestfeaturelabel = labels[bestfeature]
	mytree = {bestfeaturelabel:{}}
	labelscopy = labels[:]  #若直接将labels进行下面的del，则会在labels上操作，把labels本身改变，会影响函数外labels的使用
	del(labelscopy[bestfeature]) #删除已分配的特征
	featurevalue = [line[bestfeature] for line in dataset]
	univalue = set(featurevalue)
	for value in univalue:
		sublabels = labelscopy[:]
		newdataset = splitdataset(dataset,bestfeature,value)
		mytree[bestfeaturelabel][value] = createtree(newdataset,sublabels) # 递归
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
filename = 'lenses.txt'
fid = open(filename)
data = fid.readlines()
dataset = []
for line in data:
	line = line.strip()
	line1 = line.split('\t')[:]
	dataset.append(line1)
# dataset = [line.strip().split('\t') for line in fid.readlines()] #跟上面一样的效果，不过简洁很多
lenseslabels = ['age','prescript','astigmatic','tearrate']
lensestree = sz_c3_tree.createtree(dataset,lenseslabels)
sz_c3_treeplot.createplot(lensestree)
'''
			
		
