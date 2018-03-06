# 机器学习实战 学习记录
# Chapter 3 决策树 绘制图
# coding='UTF-8'

'''
使用matplotlib注解绘制树形图
'''
import matplotlib.pyplot as plt
# 定义文本框和箭头格式
decisionnode = dict(boxstyle = 'sawtooth',fc = '0.8') #初始化
# decisionnode = {'boxstyle':'sawtooth','fc':'0.8'}
leafnode = {'boxstyle':'round4','fc':'0.8'}
arrowargs = {'arrowstyle':'<-'}

def plotnode(nodetxt,centerpt,parentpt,nodetype):
	createplot.ax1.annotate(nodetxt,xy=parentpt,xytext=centerpt,\
	xycoords='axes fraction',textcoords='axes fraction',\
	va='center',ha='center',bbox=nodetype,arrowprops=arrowargs)
# 简单的例子
'''	
def createplot():
	fig = plt.figure(1,facecolor='white')
	fig.clf() #清除图像
	createplot.ax1 = plt.subplot(111,frameon=False)
	plotnode('a decision node',(0.5,0.1),(0.1,0.5),decisionnode)
	plotnode('a leaf node',(0.8,0.1),(0.3,0.8),leafnode)
	plt.show()
'''
# 获取叶节点的数目
def getleafnum(mytree):
	numleaf = 0
	firststr = list(mytree.keys())[0] #decisionnode 注意mytree.keys()直接得到的dict_keys类型不能直接以下标调用，需要list()一下
	secstr = mytree[firststr]
	for value in secstr.keys():
		if type(secstr[value]).__name__ == 'dict': #判断节点的数据是否为dict
			numleaf += getleafnum(secstr[value])
		else:
			numleaf += 1
	return numleaf
# 获得树的层数
def gettreedepth(mytree):
	treedepth = 0
	firststr = list(mytree.keys())[0]
	secstr = mytree[firststr]
	for value in secstr.keys():
		if type(secstr[value]).__name__ == 'dict':
			thisdepth = 1+gettreedepth(secstr[value])
		else:
			thisdepth = 1
		if thisdepth>treedepth:
			treedepth = thisdepth
	return treedepth
# 构建决策树（不用每次重新生成）
def retrievetree(i):
	listoftree = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},\
	{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
	return listoftree[i]
# mytree = sz_c3_treeplot.retrievetree(0)
# sz_c3_treeplot.getleafnum(mytree)
# sz_c3_treeplot.gettreedepth(mytree)
def plotmidtext(centerpt,parentpt,txtstr): #父子节点之间填充文本
	xmid = (parentpt[0]-centerpt[0])/2.0+centerpt[0]
	ymid = (parentpt[1]-centerpt[1])/2.0+centerpt[1]
	createplot.ax1.text(xmid,ymid,txtstr,verticalalignment='bottom',rotation=45)
# text 的参数horizontalalignment='right', 	

def plottree(mytree,parentpt,nodetxt):
	leafnum = getleafnum(mytree)
	depth = gettreedepth(mytree)
	firststr = list(mytree.keys())[0]
	# 决策点的位置：在它的所有此决策点的叶节点的中间
	# 图形页面的大小是1×1，共有几个叶节点就把宽度分为几份，每个叶节点在每份的中间；
	# 根据树的层数，将页面高度分为几层，顶点在最高位置，即高度1处，最后一层叶节点在0高度处
	centerpt = (plottree.xoff + (1.0+float(leafnum))/2.0/plottree.totalw,plottree.yoff)
	plotmidtext(centerpt,parentpt,nodetxt) # 标记子节点属性值 即箭头上的值
	plotnode(firststr,centerpt,parentpt,decisionnode)
	# 上述决策点下面的子决策点
	secstr = mytree[firststr]
	plottree.yoff = plottree.yoff - 1.0/plottree.totalD
	for value in secstr.keys():
		if type(secstr[value]).__name__ == 'dict':
			plottree(secstr[value],centerpt,str(value))
		else:
			plottree.xoff = plottree.xoff + 1.0/plottree.totalw 
			plotnode(secstr[value],(plottree.xoff,plottree.yoff),centerpt,leafnode)
			plotmidtext((plottree.xoff,plottree.yoff),centerpt,str(value))
	plottree.yoff = plottree.yoff + 1.0/plottree.totalD

def createplot(mytree):
	fig = plt.figure(1,facecolor = 'white')
	fig.clf()
	axprops = dict(xticks=[],yticks=[]) # **传入关键词参数，利用dict
	createplot.ax1 = plt.subplot(111,frameon=False,**axprops)
	plottree.totalD = float(gettreedepth(mytree))
	plottree.totalw = float(getleafnum(mytree))
	plottree.xoff = -0.5/plottree.totalw # xoff为上一个画的叶节点的横坐标，初始位置在最左端再往左半份
	plottree.yoff = 1.0 # 自顶向下画树，因此初始值在最高1处
	plottree(mytree,(0.5,1.0),'') #注意第一个决策点位于页面最高处的中间，他的箭头起始位置相同，即没有箭头画出
	plt.show()
# sz_c3_treeplot.createplot(mytree)

