# 爬虫学习与示例 1
# beautifulsoup
# https://www.indeed.com/m/jobs?q=data+scientist&l=&from=home
# 抓取相关数据
'''
beautifulsoup安装：
pip install beautifulsoup4
'''
from bs4 import BeautifulSoup

import urllib.request
import re

# 找到单个jod的link
url = 'https://www.indeed.com/m/jobs?q=data+scientist&l=&from=home' 
# 初始页
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page,'lxml') # 解析页面，通常页面可以使用‘lxml’解析
# 找出所有tag为a，attrs={‘rel’:[‘nofollow’]} 的信息
all_match = soup.findAll('a',attrs={'rel':['nofollow']})
# 打印出所有信息
for i in all_match:
	print(i['href'])
	print(type(i['href']))
	print(r'https://www.indeed.com/m/%s' % (i['href']))
# 打开每个job的链接，找到其中职位的描述
f = open('1.txt','w')
for each in all_match:
	jd_url = 'https://www.indeed.com/m/'+each['href']
	jd_page = urllib.request.urlopen(jd_url)
	jd_soup = BeautifulSoup(jd_page,'lxml')
	# 查找job描述
	# title不需要re
	title = jd_soup.html.head.title
	
	f.write(title.string+'\n')
	
	jd_desc = jd_soup.findAll('div',attrs={'id':['desc']})
	f.write(str(jd_desc))
	f.write('\n#########################################\n')
	
	# print(jd_desc)
	# 文本清洗，使用 re 或者 str.replace
	
	
	
	
	
	
	



