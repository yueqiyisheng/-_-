# coding='UTF-8'

'''
问题描述：
给定一个字符串s和一个字符串列表的字典，如果字符串包含字典中的字符串，
则用<b>和</b>标记，如果不同的字符串重叠，需要进行合并
例1：s = 'abcxyz123' dict = ['abc','123']
输出：'<b>abc</b>xyz<b>123</b>'

例2：s = 'aaabbcc'  dict = ['aaa','aab','bc']
输出：'<b>aaabbc</b>c'
'''

# 思路：利用tag先标记s中字符是否满足在dict中的条件，这样重复也只是重复标记为1；
#       然后，根据tag前后关系加<b>和</b>
# 注意首尾

# class Solution(object):
def add(s,dict):
	tag = list(zeros((len(s))))
	for item in dict:
		start = 0
		start2 = 0
		while(start2<len(s) and start>=0):
			# s中可能有多个item，需要都find
			# 从s的start2位置开始find item，返回的是s中的index
			start = s.find(item,start2) 
			start2 = start + len(item)
			if start>=0: #没有找到，返回的是-1
				for i in range((len(item))):
					tag[start+i] = 1 #依次将tag标记
	ns = []
	if tag[0] == 1:
		ns.append('<b>')
	ns.append(s[0])
	for i in range(((len(s))-1)):
		if tag[i]==0 and tag[i+1] == 1: #开始
			ns.append('<b>')
		elif tag[i]==1 and tag[i+1]==0: #结束
			ns.append('</b>')
		ns.append(s[i+1])
	if tag[len(s)-1] ==1:
		ns.append('</b>')
	return ''.join(ns) #ns是list，转换成str
	
