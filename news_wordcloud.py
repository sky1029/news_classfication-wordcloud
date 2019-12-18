# -*-coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud
from itertools import chain

#加载文件
news = pd.read_csv("chinese_news.csv")

#查看文件内容
print(news.head())
print(news.info())

#用标题填充内容为空的行
index = news[news["content"].isnull()].index
news["content"][index] = news["headline"][index]

#重复值处理
print(news.duplicated().sum())
news.drop_duplicates(inplace=True)


#去掉标点符号和停用词
def get_stopword():
	s = set()
	with open("stop_words.txt", encoding="UTF-8") as f:
		for line in f:
			s.add(line.strip())
	return s

def remove_stopwords(text):
	for stopword in stopwords:
		text = text.replace(stopword, "")
	return text

stopwords = get_stopword()
news["content"] = news["content"].apply(remove_stopwords)

#分词
def cut_word(text):
	return jieba.cut(text)

news["content"] = news["content"].apply(cut_word)


#处理内容列文本格式，转换成由空格连接的字符串
#将分词后得到的生成器转成列表
def get_wordlist(text):
	return [word for word in text]
news["content"] = news["content"].apply(get_wordlist)
#将所有content串联成一个list
words= list(chain(*news["content"]))
words = " ".join(words)


#生成词云
wc = WordCloud(font_path="STHeiti Light.ttc",
				background_color="white",
				mask=plt.imread("map.jpg"),
				max_words=100,
				width=1000,
				height=800)
img = wc.generate(words)
plt.figure(figsize=(25,20))
plt.imshow(img)
plt.axis("off")
wc.to_file("wordcloud.jpg")




