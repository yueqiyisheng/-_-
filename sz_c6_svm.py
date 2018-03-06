# 机器学习实战 学习记录
# Chapter 6 支持向量机 SVM 
# SVM本身是二分类问题，对多类问题需要修改代码
# coding='UTF-8'

'''
基于最大间隔进行分类
思路：先找到具有最小间隔的数据点，然后对该间隔进行最大化
'''
'''
SVM的主要任务就是找到那些使得目标函数成立的alpha
SMO（序列最小化，坐标上升法的进步版）就是完成上述任务的算法，即主要来训练SVM
1）简化版SMO：选择alpha对时，首先在数据集上遍历每一个alpha，然后在剩下的alpha集合中随机选择另一个alpha
2）完整版：外循环确定要优化的最佳alpha对
'''

# 简化版SMO

## SMO辅助函数

#读入数据
from numpy import *
def loaddataset(filename):
	fr = open(filename)
	dataset = [];labelset = []
	for line in fr.readlines():
		items = line.strip().split('\t')
		dataset.append([float(items[0]),float(items[1])])
		labelset.append(float(items[2]))
	return dataset,labelset
# 从一定范围内随机选择一个整数
def selectrand(i,m): # 从0到m中选择一个不是i的整数【用于简化版SMO挑选第二个alpha】的下标
	j=i
	while j==i:
		j=int(random.uniform(0,m))
	return j
# alpha有范围约束，0<alpha<C
def clipalpha(aj,H,L):
	if aj>H: aj=H
	elif aj<L: aj=L
	return aj
'''
简化版SMO伪代码：

创建一个alpha向量，将其初始化为0向量
当迭代次数小于最大迭代次数（外循环）：
	对数据集中的每个数据向量（内循环）：【？什么数据向量】
		如果该数据向量可以被优化：
			随机选择另外一个数据向量
			同时优化这两个向量
		    如果这两个向量均不能被优化，退出内循环
	如果所有向量都没被优化，增加迭代次数，继续下一次循环
'''
# 简化版SMO算法【？？？理论推导？？】
def amosimple(dataset,labelset,C,toler,maxiter): #输入数据集、标签、常数C、toler容错率、最大迭代次数
	datamat = mat(dataset);labelmat = mat(labelset).transpose() #转换成numpy矩阵 labelmat为列向量
	b=0;m,n=shape(datamat)
	alpha = mat(zeros((m,1))) # 初始化alpha向量 m×1的列向量 m为训练样本个数
	iter = 0 # 存储没有任何alpha改变的情况下遍历数据集的次数？
	while iter < maxiter: # 外循环
		alphapairchanged = 0 # 记录alpha是否进行优化
		for i in range(m): # 数据集顺序遍历
			
			fxi = float(multiply(alpha,labelmat).T*(datamat*datamat[i,:].T))+b # 预测的类别 f(xi)=w.T * xi + b ; w = ∑αyx w是列向量n×1
			ei = fxi - float(labelmat[i]) # 预测分类结果与正确结果的误差
			# yi(w.T*x+b) -1 与toler的关系 与 alpha与C的关系？？？？？
			if ((labelmat[i]*ei < -toler) and (alpha[i]<C)) or \
			((labelmat[i]*ei>toler) and (alpha[i]>0)):
			# 判断该alpha是否需要优化：如果误差很大（超出容错范围），则对该数据实例对应的alpha值进行优化
			# 同时检查alpha值，保证其不等于0或者C，因为0或者C的话，就已经在“边界”上了，不能再减小或增大，即不值得再对其优化
			# 【问题】：这两个 and 的关系 为什么是这样的？？
				# 随机选择另一个alpha
				j = selectrand(i,m)  
				fxj = float(multiply(alpha,labelmat).T*(datamat*datamat[j,:].T))+b
				ej = fxj - labelmat[j] # 误差
				alphaiold = alpha[i].copy(); alphajold = alpha[j].copy() # 保留原alpha值
				# 保证 alpha_j 在0到C之间
				if (labelmat[i] != labelmat[j]):
					L = max(0,alpha[j]-alpha[i])
					H = min(C,C+alpha[j]-alpha[i])
				else:
					L = max(0,alpha[j]+alpha[i]-C)
					H = min(C,alpha[j]+alpha[i])
				if L == H: print('L = H');continue # 退出此次循环，直接进行下一次for循环
				# eta是 alpha_j 的最优修改量
				eta = 2.0*datamat[i,:]*datamat[j,:].T-datamat[i,:]*datamat[i,:].T-datamat[j,:]*datamat[j,:].T # 2xixj-xi^2-xj^2
				if eta >= 0: print('eta>=0');continue
				alpha[j] -= labelmat[j]*(ei - ej)/eta
				alpha[j] = clipalpha(alpha[j],H,L)
				if (abs(alpha[j]-alphajold) < 0.00001): print('j not moving enough'); continue # alpha_j 改变很小，则退出for循环
				alpha[i] += labelmat[j]*labelmat[i]*(alphajold-alpha[j]) # alpha_j 与 alpha_i 改变量相同，但方向相反
				b1 = b-ei-labelmat[i]*(alpha[i]-alphaiold)*datamat[i,:]*datamat[i,:].T - labelmat[j]*(alpha[j]-alphajold)*datamat[i,:]*datamat[j,:].T
				b2 = b-ej-labelmat[i]*(alpha[i]-alphaiold)*datamat[i,:]*datamat[j,:].T - labelmat[j]*(alpha[j]-alphajold)*datamat[j,:]*datamat[j,:].T
				if (0 < alpha[i]) and (C > alpha[i]): b=b1
				elif (0 < alpha[j]) and (C > alpha[j]): b= b2
				else: b =(b1+b2)/2.0
				# 若程序执行到此没有执行continue，则说明已经成功改变了一对alpha
				alphapairchanged += 1
				print('iter: %d i: %d, pairs changed %d' % (iter,i,alphapairchanged))
		if alphapairchanged == 0: iter +=1
		else: iter = 0 # 只有在所有数据集上遍历了iter次。且不再发生任何alpha修改，退出while循环，
		print('iteration number: %d' % (iter))
	return b,alpha
'''
import time
start = time.clock()
filename = r'.\data\testSet.txt'
dataset,labelset = sz_c5_svm.loaddataset(filename)
b,alpha = sz_c5_svm.amosimple(dataset,labelset,0.6,0.001,40)
end = time.clock()
print('running time: %s seconds.' % (end-start))
'''

# alpha中大多数为0，仅有几个是不为0的数    alpha[alpha>0] 仅对numpy类型有用的过滤实例
# 不为0的alpha个数即为支持向量的个数

'''
完整版与简化版的主要不同：
（1）第一个alpha的选取：在两种方式之间交替，① 是在所有数据集上进行单遍扫描；② 是在非边界【即不为0或C】alpha中实现单遍扫描
	②中需要建立非边界alpha的列表，同时跳过那些已知不会改变的alpha值
（2）第二个alpha的选取：通过“最大化步长”来选取，建立一个全局的缓存来保存alpha_j 对应的误差值，从中选择使得步长或者说ei-ej最大的值
'''

# 完整版SMO的支持函数
# 作为一个数据结构来使用对象，将值传给函数时，可以通过将所有数据移到一个结构中实现
# 全局的一个结构，函数之间不需要传
class optstruct:
	def __init__(self,datamat,labelmat,C,toler,ktup):
		self.x = datamat
		self.label = labelmat
		self.c = C
		self.tol = toler
		self.m = list(shape(datamat))[0]
		self.alpha = mat(zeros((self.m,1)))
		self.b = 0
		self.ecache = mat(zeros((self.m,2))) # 误差缓存初始化 第1列标记是否有效，第二列为实际e值
		self.K = mat(zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i] = kerneltrans(self.x,self.x[i,:],ktup) #每个样例，都需要与其他和自己所有样例进行kernel里的运算得到一个列向量
	
# 计算误差ek		
def calcek(os,k):
	fxk = float(multiply(os.alpha,os.label).T*(os.x*os.x[k,:].T))+os.b
	ek = fxk - float(os.label[k])
	return ek
# 内循环中的启发式方法
def selectj(i,os,ei):
	maxk = -1; maxdeltae = 0; ej = 0
	os.ecache[i] = [1,ei]
	validecachelist = nonzero(os.ecache[:,0].A)[0] #.A是将matrix转换为array 返回的是下标
	if (len(validecachelist)) > 1:
		for k in validecachelist:
			if k == i: continue
			# ek = calcek(os,k)
			ek = calcekk(os,k)
			deltae = abs(ei-ek)
			if (deltae > maxdeltae):
				maxk = k; maxdeltae = deltae; ej = ek
		return maxk,ej
	else: # 第一次循环，则随机选择一个j值
		j = selectrand(i,os.m)
		# ej = calcek(os,j)
		ej = calcekk(os,j)
	return j,ej

def updateek(os,k):
	# ek = calcek(os,k)
	ek = calcekk(os,k)
	os.ecache[k] = [1,ek] #对alpha进行优化更新ek

# 内循环：alpha的优化
def innerl(i,os):
	# ei = calcek(os,i)
	ei = calcekk(os,i)
	if ((os.label[i]*ei < -os.tol) and (os.alpha[i] < os.c)) or ((os.label[i]*ei > os.tol) and (os.alpha[i] > 0)):
		j,ej = selectj(i,os,ei)
		alphaiold = os.alpha[i].copy(); alphajold = os.alpha[j].copy(); # 这里需要使用copy 不用的话并没有真正将原来的值保存下来，只是多了一个此位置的标签
		if (os.label[i] != os.label[j]):
			L = max(0,os.alpha[j] - os.alpha[i])
			H = min(os.c, os.c+os.alpha[j] - os.alpha[i])
		else:
			L = max(0,os.alpha[j] + os.alpha[i] - os.c)
			H = min(os.c, os.alpha[j] + os.alpha[i])
		if L == H: print('L=H'); return 0
		eta = 2.0*os.x[i,:]*os.x[j,:].T-os.x[i,:]*os.x[i,:].T-os.x[j,:]*os.x[j,:].T # 2xixj-xi^2-xj^2
		if eta >= 0: print('eta>=0');return 0
		os.alpha[j] -= os.label[j]*(ei-ej)/eta
		os.alpha[j] = clipalpha(os.alpha[j],H,L)
		updateek(os,j) # alpha更新后e也会变化，需要随之更新
		if (abs(os.alpha[j]-alphajold) < 0.00001): 
			print('j not moving enough'); return 0
		os.alpha[i] += os.label[j]*os.label[i]*(alphajold-os.alpha[j]) # alpha_j 与 alpha_i 改变量相同，但方向相反
		b1 = os.b-ei-os.label[i]*(os.alpha[i]-alphaiold)*os.x[i,:]*os.x[i,:].T - os.label[j]*(os.alpha[j]-alphajold)*os.x[i,:]*os.x[j,:].T
		b2 = os.b-ej-os.label[i]*(os.alpha[i]-alphaiold)*os.x[i,:]*os.x[j,:].T - os.label[j]*(os.alpha[j]-alphajold)*os.x[j,:]*os.x[j,:].T		
		if (0 < os.alpha[i]) and (os.c > os.alpha[i]): os.b=b1
		elif (0 < os.alpha[j]) and (os.c > os.alpha[j]): os.b= b2
		else: os.b =(b1+b2)/2.0
		return 1
	else: return 0

# 外循环
def smoP(dataset,labelset,C,toler,maxiter,ktup=('lin',0)):
	os = optstruct(mat(dataset),mat(labelset).transpose(),C,toler,ktup) # 将数据存为 class 数据结构中
	iter = 0
	entireset = True # 
	alphapairchanged = 0
	while (iter < maxiter) and ((alphapairchanged >0) or (entireset)):
	# 该进行遍历所有值时，alphapairchanged可以是0（表示循环刚开始或者非边界值中没有可以改变的了）
	# 但是，当该进行非边界值遍历时，alphapairchanged 为0，表明上一次是遍历所有值却没有需要改变的alpha值，则此时退出循环
		alphapairchanged = 0
		if entireset: # 遍历所有的值
			for i in range(os.m):
				# alphapairchanged += innerl(i,os) #改变的对数
				alphapairchanged += innerlk(i,os) # 使用核函数
				print('fullset, iter: %d i: %d, pairs changed %d' % (iter,i,alphapairchanged))
			iter += 1
		else: # 遍历非边界的值
			nonbound = nonzero((os.alpha.A > 0)*(os.alpha.A < C))[0] 
			for i in nonbound:
				# alphapairchanged += innerl(i,os)
				alphapairchanged += innerlk(i,os) # 使用核函数
				print('non-bound, iter: %d, i: %d, pairs changed %d' % (iter,i,alphapairchanged))
			iter += 1
		if entireset: entireset = False  # 表明刚刚遍历过一次所有的值
		elif (alphapairchanged == 0): entireset = True # 遍历非边界的值后，没有改变的alpha值，则重新开始遍历所有的值
		# 遍历完一次所有的值后，开始重复遍历非边界值，直至非边界值中没有alpha再改变，重新开始遍历所有的值
		print('iteration number: %d' % (iter))
	return os.b,os.alpha
'''
import time
start = time.clock()
filename = r'.\data\testSet.txt'
dataset,labelset = sz_c5_svm.loaddataset(filename)
b,alpha = sz_c5_svm.smoP(dataset,labelset,0.6,0.001,40)
end = time.clock()
print('running time: %s seconds.' % (end-start))	
'''	
# 常数C一方面保证所有样例的间隔不小于1.0；一方面要使得分类间隔尽可能大；两方平衡
# 若C很大，则分类器尽可能将所有样例分类正确	

# datamat = mat(dataset);labelmat = mat(labelset).transpose()
# w = multiply(alpha,labelmat).T*datamat #超平面参数
	
def calw(alpha,dataset,labelset):
	datamat = mat(dataset);labelmat = mat(labelset).transpose()
	m,n = shape(datamat)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(labelmat[i]*alpha[i],datamat[i,:].T)
	# 等效于：
	# w = (multiply(alpha,labelmat).T*datamat).T
	return w
# w1 = sz_c5_svm.calw(alpha,dataset,labelset)
			
# 核转换函数
def kerneltrans(x,a,ktup): #输入的是矩阵形式 ktup=('lin',0)这种形式，第一个指示核函数的形式，第二个是sigma的值
	m,n = shape(x)
	k = mat(zeros((m,1)))
	if ktup[0]=='lin': k = x*a.T #线性
	elif ktup[0]=='rbf':  #高斯
		for j in range(m):
			deltar = x[j,:] - a
			k[j] = deltar*deltar.T
		k = exp(k/(-1*ktup[1]**2))
	else: raise NameError('Houston We Have a Problem: the Kerlnel is not recognized')
	return k
# 修改下列两个函数内的部分语句，适用于使用kernel
# !!!注意将其他引用此函数的地方改了！！！
def innerlk(i,os):
	# ei = calcek(os,i)
	ei = calcekk(os,i)
	if ((os.label[i]*ei < -os.tol) and (os.alpha[i] < os.c)) or ((os.label[i]*ei > os.tol) and (os.alpha[i] > 0)):
		j,ej = selectj(i,os,ei)
		alphaiold = os.alpha[i].copy(); alphajold = os.alpha[j].copy(); # 这里需要使用copy 不用的话并没有真正将原来的值保存下来，只是多了一个此位置的标签
		if (os.label[i] != os.label[j]):
			L = max(0,os.alpha[j] - os.alpha[i])
			H = min(os.c, os.c+os.alpha[j] - os.alpha[i])
		else:
			L = max(0,os.alpha[j] + os.alpha[i] - os.c)
			H = min(os.c, os.alpha[j] + os.alpha[i])
		if L == H: print('L=H'); return 0
		eta = 2.0*os.K[i,j]-os.K[i,i]-os.K[j,j] # 2xixj-xi^2-xj^2  # os.K
		if eta >= 0: print('eta>=0');return 0
		os.alpha[j] -= os.label[j]*(ei-ej)/eta
		os.alpha[j] = clipalpha(os.alpha[j],H,L)
		updateek(os,j) # alpha更新后e也会变化，需要随之更新
		if (abs(os.alpha[j]-alphajold) < 0.00001): 
			print('j not moving enough'); return 0
		os.alpha[i] += os.label[j]*os.label[i]*(alphajold-os.alpha[j]) # alpha_j 与 alpha_i 改变量相同，但方向相反
		b1 = os.b-ei-os.label[i]*(os.alpha[i]-alphaiold)*os.K[i,i] - os.label[j]*(os.alpha[j]-alphajold)*os.K[i,j]
		b2 = os.b-ej-os.label[i]*(os.alpha[i]-alphaiold)*os.K[i,j] - os.label[j]*(os.alpha[j]-alphajold)*os.K[j,j]	
		if (0 < os.alpha[i]) and (os.c > os.alpha[i]): os.b=b1
		elif (0 < os.alpha[j]) and (os.c > os.alpha[j]): os.b= b2
		else: os.b =(b1+b2)/2.0
		return 1
	else: return 0
	
def calcekk(os,k):
	fxk = float(multiply(os.alpha,os.label).T*os.K[:,k]+os.b)
	ek = fxk - float(os.label[k])
	return ek	

def testrbf(k1 = 1.3): #输入的是高斯核中的sigma
	filename = r'.\data\testSetRBF.txt'
	dataset,labelset = loaddataset(filename)
	b,alpha = smoP(dataset,labelset,200,0.0001,10000,('rbf',k1))
	datamat = mat(dataset);labelmat = mat(labelset).transpose()
	# 构建支持向量矩阵
	svind = nonzero(alpha.A > 0)[0] #支持向量对应的下标
	svs = datamat[svind] # 提取出支持向量
	svlabel = labelmat[svind]
	print('there are %d support vectors' % (shape(svind)[0]))
	m,n = shape(datamat)
	errorcount = 0
	for i in range(m):
		ker = kerneltrans(svs,datamat[i,:],('rbf',k1)) # 仅仅对于支持向量来计算
		predict = ker.T*multiply(svlabel,alpha[svind])+b
		if sign(predict) != sign(labelset[i]): errorcount += 1
	print('the training error rate is: %f' % (float(errorcount/m)))
	# 测试
	filename = r'.\data\testSetRBF2.txt'
	dataset,labelset = loaddataset(filename)
	datamat = mat(dataset);labelmat = mat(labelset).transpose()
	errorcount = 0
	m,n = shape(datamat)
	for i in range(m):
		ker = kerneltrans(svs,datamat[i,:],('rbf',k1)) # 仅仅对于支持向量来计算
		predict = ker.T*multiply(svlabel,alpha[svind])+b
		if sign(predict) != sign(labelset[i]): errorcount += 1
	print('the testing error rate is: %f' % (float(errorcount/m)))
'''
示例：手写识别问题
KNN每次需要使用所有训练样本来分类，占用内存过大，而SVM只需要将支持向量保存即可
'''
# 数据转换成向量：需要将图像的矩阵数据（32*32）转换成一个向量1024
def img2vector(filename):
	vec = zeros((1,1024))
	fr = open(filename)
	#依次读取前32行数据，并将每行的前32个字符存在向量中
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			vec[0,32*i+j] = int(line[j]) #不能将一行32个一起int，然后传过去
	return vec
# 读入训练样本
def loadimage(dirname): #文件所在文件夹路径
	from os import listdir,path 
	hwlabel = []
	trainfilelist = listdir(dirname)
	# 读取文件夹中的目录内容 注意 r 
	m = len(trainfilelist)
	dataset = zeros((m,1024))
	# 生成训练数据集
	for i in range(m):
		# 从文件名中解析中分类数字
		filenamestr = trainfilelist[i]
		classnum = int(filenamestr[0]) #classnum = int((filenamestr.split('.')[0]).split('_')[0])
		if classnum == 9: hwlabel.append(-1)
		else: hwlabel.append(1)
		# 本质上SVM是二分类，因此此处仅将9和其他分类
		filename = path.join(dirname,filenamestr)
		dataset[i,:] = img2vector(filename)
	return dataset,hwlabel
# 训练并测试分类器
def testdigit(ktup=('rbf',10)):
	dataset,labelset = loadimage(r'trainingDigits')
	b,alpha = smoP(dataset,labelset,200,0.0001,10000,ktup)
	datamat = mat(dataset);labelmat = mat(labelset).transpose()
	# 构建支持向量矩阵
	svind = nonzero(alpha.A > 0)[0] #支持向量对应的下标
	svs = datamat[svind] # 提取出支持向量
	svlabel = labelmat[svind]
	print('there are %d support vectors' % (shape(svind)[0]))
	m,n = shape(datamat)
	errorcount = 0
	for i in range(m):
		ker = kerneltrans(svs,datamat[i,:],ktup) # 仅仅对于支持向量来计算
		predict = ker.T*multiply(svlabel,alpha[svind])+b
		if sign(predict) != sign(labelset[i]): errorcount += 1
	print('the training error rate is: %f' % (float(errorcount/m)))
	# 测试
	dataset,labelset = loadimage(r'testDigits')
	datamat = mat(dataset);labelmat = mat(labelset).transpose()
	m,n = shape(datamat)
	errorcount = 0
	for i in range(m):
		ker = kerneltrans(svs,datamat[i,:],ktup) # 仅仅对于支持向量来计算
		predict = ker.T*multiply(svlabel,alpha[svind])+b
		if sign(predict) != sign(labelset[i]): errorcount += 1
	print('the test error rate is: %f' % (float(errorcount/m)))
# 之后的重点是要搞懂如何选择sigma值和C值！！！！	
