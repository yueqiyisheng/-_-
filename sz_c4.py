# 机器学习实战 学习记录
# Chapter 4 基于概率论的分类方法：朴素贝叶斯
# coding='UTF-8'
'''
优 点 ：在数据较少的情况下仍然有效，可以处理多类别问题。
缺 点 ：对于输入数据的准备方式较为敏感。
适用数据类型：标称型数据
'''
'''
使用朴素贝叶斯进行文档分类
朴素：意味着两个假设：
1. 特征的独立性，对于文本来说即：一个特征或单词出现的可能性与其他单词没有关系
2. 每个特征同等重要
这两个假设实际中均存在问题，但实际效果不错
'''
# 从文本中构建词向量：
# 考虑所有文档中出现的单词，决定将那些放入到词汇表中；
# 将每篇文档转换为词汇表上的向量

# 简单的示例1：鱼分类
def loaddataset():
	postinglist = [['my','dog','has','flea','problem','help','please'],
				   ['maybe','not','take','him','to','dog','park','stupid'],
				   ['my','dalmation','is','so','cute','I','love','him'],
				   ['stop','posting','stupid','worthless','garbage'],
				   ['mr','licks','ate','my','steak','how','to','stop','him'],
				   ['guit','buying','worthless','dog','food','stupid']]
	classvec = [0,1,0,1,0,1] # 1 表示侮辱性文字，0 表示正常言论
	return postinglist,classvec
# 创建词汇表
def createvocablist(dataset):
	vocabset = set([]) #创建一个空集
	for document in dataset:
		vocabset = vocabset | set(document) # 取并集 所有词不重复
	return list(vocabset)  # 集合不能以下标调用，需要将其转换为list
# 将输入文档转换为词汇表上的向量
def set2vec(vocablist,inputset):
	inputvec = [0]*len(vocablist) #创建一个全为0的list
	for item in inputset:
		if item in vocablist: # 对于输入文本的每个词，检查其是否再字典中，若有，则把相应位置置1
			inputvec[vocablist.index(item)] = 1
		else:
			print('the word: #s is not in my vocabulary!' % (item))
	return inputvec
# postinglist,classvec = sz_c4.loaddataset()
# vocabset = sz_c4.createvocablist(postinglist)
# sz_c4.set2vec(vocabset,postinglist[0])

# 训练算法：从词向量计算概率
'''
伪代码：??
计算每个类别中的文档数目
对每篇训练文档：
	对每个类别：
		如果词条出现文档中―增加该词条的计数值
		增加所有词条的计数值
	对每个类别：
		对每个词条：
			将该词条的数目除以总词条数目得到条件概率
	返回每个类别的条件概率
'''
from numpy import *
def trainNB0(trainset,trainlabel): # 输入 已经映射到词汇表的训练样本，和训练样本对应的分类标签（标称型数值 0 1）
	numtraindoc = len(trainset) # 训练样本数量
	numword = len(trainset[0]) # 词汇表长度
	pc1 = sum(trainlabel)/float(numtraindoc) # 1对应类别的先验概率p(c1) 【针对二分类问题，多分类需要修改】
	# p0num = zeros(numword) # 初始化，计算词汇表中每个词在两个类别中出现的次数-频率
	# p1num = zeros(numword)
	p0num = ones(numword) #考虑到概率的乘积中若一个为0，则最后结果为0，因此所有词出现的次数初始化为1
	p1num = ones(numword)
	p0denom = 2.0 # 分母初始化为2？这样所有概率相加和不等于1？？
	p1denom = 2.0 # 我觉得书上设置为2是错误的，应该是所有可取值的个数，即c1类别中出现的词条类型个数
	for i in range(numtraindoc):
		if trainlabel[i] == 1: #判断每个文档属于哪个类别
			p1num = p1num + trainset[i] #对于仅统计词条是否存在的模型，和 统计词条出现次数的模型 均适用
			p1denom += sum(trainset[i]) 
		else:
			p0num = p0num + trainset[i]
			p0denom += sum(trainset[i])
	# p0vec = p0num/p0denom
	# p1vec = p1num/p1denom
	# p0vec = log(p0num/p0denom) #很多很小的数相乘会引起“下溢出”，因此取自然对数
	# p1vec = log(p1num/p1denom)
	p0vec = log(p0num/sum(p0num))
	p1vec = log(p1num/sum(p1num))
	return p0vec,p1vec,pc1
'''                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
postinglist,classvec = sz_c4.loaddataset()
vocabset = sz_c4.createvocablist(postinglist)
trainset = []
for doc in postinglist:
	trainset.append(sz_c4.set2vec(vocabset,doc))
p0vec,p1vec,pc1 = sz_c4.trainNB0(trainset,classvec)
'''
def classifynb(vec2classify,p0vec,p1vec,pc1):
	p0 = sum(vec2classify * p0vec)+log(1.0-pc1)
	p1 = sum(vec2classify * p1vec)+log(pc1)
	if p0>p1:
		return 0
	else:
		return 1
'''		
def testnb():
	postinglist,classvec = loaddataset()
	vocabset = createvocablist(postinglist)
	trainset = []
	for doc in postinglist:
		trainset.append(set2vec(vocabset,doc))
	p0vec,p1vec,pc1 = trainNB0(trainset,classvec)
	testinput = ['love','my','dalmation']
	testvec = set2vec(vocabset,testinput)
	print(testinput,'classified as:',classifynb(testvec,p0vec,p1vec,pc1))
	testinput = ['stupid','garbage']
	testvec = set2vec(vocabset,testinput)
	print(testinput,'classified as:',classifynb(testvec,p0vec,p1vec,pc1))
'''
'''
一个词在文档中出现的次数在上述方法中没考虑；
考虑出现次数：“词袋模型”
'''
def bagset2vec(vocablist,inputset):
	inputvec = [0]*len(vocablist) #创建一个全为0的list
	for item in inputset:
		if item in vocablist: # 对于输入文本的每个词，检查其是否再字典中，若有，则把相应位置置1
			inputvec[vocablist.index(item)] += 1
		else:
			print('the word: %s is not in my vocabulary!' % (item))
	return inputvec
'''
示例2：使用朴素贝叶斯过滤垃圾邮件
'''
# 准备数据：切分文本，即将文档转换成词条
# 使用正则表达式切分文本；
# 注意当文本中有URL、HTML等网址对象时，需要用更高级的过滤器来解析；
# 文本解析是个复杂的过程，本例仅是极其简单的情况 

def textparse(fulltext):
	import re # 正则表达式
	temp = re.split(r'\W*',fulltext)
	return [tok.lower() for tok in temp if len(tok)>2] # 长度大于2 是考虑到URL和HTML中各种小字符串

def spamtest():
	doclist=[]; classlist=[]; fulltext=[]
	# 每个文件夹中各有25份邮件
	for i in range(1,26): #从1到25
		# print(i)
		emailtext = open(r'.\email\ham\%d.txt' % (i)).read()
		wordlist = textparse(emailtext) #词条
		doclist.append(wordlist)
		classlist.append(0)
		fulltext.extend(wordlist) # 用处？
		emailtext = open(r'.\email\spam\%d.txt' % (i)).read()
		wordlist = textparse(emailtext) #词条
		doclist.append(wordlist)
		classlist.append(1) #1:垃圾邮件
		fulltext.extend(wordlist) # 用处？
	vocablist = createvocablist(doclist)
	# 从共50份邮件中随机选取一部分构建训练集 和 测试集（10份）
	# 留存交叉验证 为了更精确地估计错误率，应该进行多次迭代求平均错误率
	trainlist = list(range(50));testlist = []
	for i in range(10):
		randindex = int(random.uniform(0,len(trainlist))) #随机产生训练集中的一个下标
		testlist.append(trainlist[randindex])
		del(trainlist[randindex])
	trainset = []; trainclass = []
	# 将各个词条转换成词汇表上的向量
	for index in trainlist:
		trainset.append(set2vec(vocablist,doclist[index]))
		trainclass.append(classlist[index])
	p0vec,p1vec,pc1 = trainNB0(trainset,trainclass)
	# 测试分类器，计算错误率
	errorcount = 0
	for index in testlist:
		classified = classifynb(set2vec(vocablist,doclist[index]),p0vec,p1vec,pc1)
		if classified != classlist[index]:
			errorcount += 1
			print('error classified email:',doclist[index],'true class is:',classlist[index])
	print('the error rate is: %f.' % (errorcount/float(len(testlist))))
	return errorcount/float(len(testlist))
# 注意将垃圾邮件误判为正常邮件要比将正常邮件误判为垃圾邮件好，为避免错误，需要修正分类器——》第7章
'''
示例3：使用朴素贝叶斯分类器从个人广告中获取区域倾向
'''	
def calmostfreq(vocablist,fulltext):
	import operator
	freqdict = {}
	for word in vocablist:
		freqdict[word] = fulltext.count(word) #count list中word出现的次数
	sortedfreq = sorted(freqdict.items(),key=operator.itemgetter(1),reverse=True)
	return sortedfreq[:30]

def localword(feed1,feed0):
	import feedparser
	doclist = [];classlist=[];fulltext=[]
	minlen = min(len(feed0['entries']),len(feed1['entries']))
	for i in range(minlen):
		wordlist = textparse(feed1['entries'][i]['summary']) # 类似于一份邮件的全部文本,然后解析成词条
		doclist.append(wordlist)
		classlist.append(1)
		fulltext.extend(wordlist)
		wordlist = textparse(feed0['entries'][i]['summary']) # 类似于一份邮件的全部文本,然后解析成词条
		doclist.append(wordlist)
		classlist.append(0)
		fulltext.extend(wordlist)
	vocablist = createvocablist(doclist)
	top30word = calmostfreq(vocablist,fulltext) #出现频次最高的30个词 及其 次数 【dict】
	for pairw in top30word:
		if pairw[0] in vocablist:
			vocablist.remove(pairw[0]) # 从词汇表中去除这些高频词汇
	'''
	移除高频词的原因：很多词汇是冗余和结构辅助性内容，很少的词会占用所有用词的很大比率。
	除了移除高频词，还可以使用“停用词表”：从某个预定词表中移除结构上的辅助词
	'''
	# 随机抽出一部分作为训练集，一部分作为测试集
	trainlist = list(range(2*minlen));testlist = []
	for i in range(20): #20个作为测试集
		randindex = int(random.uniform(0,len(trainlist)))
		testlist.append(trainlist[randindex]) #注意：trainset大小一直变化，其中存的数值不再等于变化后的下标
		del(trainlist[randindex])
	trainset = [];trainclass=[]
	for index in trainlist:
		trainset.append(bagset2vec(vocablist,doclist[index])) #这里使用词袋模型，即统计词条在一个文本中出现的次数
		trainclass.append(classlist[index])
	p0vec,p1vec,pc1 = trainNB0(trainset,trainclass)
	errorcount = 0
	for index in testlist:
		classified = classifynb(bagset2vec(vocablist,doclist[index]),p0vec,p1vec,pc1)
		if classlist[index] != classified:
			errorcount += 1
	print('the error rate is: %f' % (float(errorcount/len(testlist))))
	return vocablist,p0vec,p1vec
	
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocablist,psf,pny = sz_c4.localword(ny,sf)

# 分析得到的两个地域的各词条使用频率数据，得到显示区域相关的用词
def gettopword(ny,sf):
	import operator
	vocablist,psf,pny = localword(ny,sf)
	topny = [];topsf=[] 
	for i in range(len(psf)):
		if psf[i]>-6.0: topsf.append((vocablist[i],psf[i])) # 先选出概率大于一定值的词条
		if pny[i]>-6.0: topny.append((vocablist[i],pny[i]))
	sortedsf = sorted(topsf,key = lambda pair:pair[1],reverse=True) # 再将词条按照频率排序，输出
	# 注意key的使用：指示如何从要比较的对象中获得比较的标准，例如，从dict中获得值；从tuple中获得第二项等
	# 指示的是获得的方式（即类似函数的形式），而不是具体的对象，如a[1]
	print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
	for item in sortedsf:
		print(item[0])
	sortedny = sorted(topny,key = lambda pair:pair[1],reverse=True) 
	print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
	for item in sortedny:
		print(item[0])
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')	
# sz_c4.gettopword(ny,sf)
	





