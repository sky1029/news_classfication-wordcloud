# 新闻分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import metrics


# 加载数据集
news = pd.read_csv("chinese_news.csv")

#数据探索
print(news.head())
print(news.info())

#数据清洗
#用标题填充内容为空的行
index = news[news["content"].isnull()].index
news["content"][index]=news["headline"][index]

#查看是否有重复的行
print(news.duplicated().sum())

#删除重复行
news.drop_duplicates(inplace=True)

#将内容列进行分词
def cut_word(text):
	return jieba.cut(text)

news["content"] = news["content"].apply(cut_word)

#去掉内容中的停用词，停用词表中包含标点符号
#加载停用词表
def get_stopword():
	s = set()
	with open("stop_words.txt", encoding="UTF-8") as f:
		for line in f:
			s.add(line.strip())
	return s
stopword = get_stopword()

#去除停用词
def remove_stopword(words):
	return [word for word in words if word not in stopword]

news["content"] = news["content"].apply(remove_stopword)

#内容和分类列格式转换
def join(text_list):
	return " ".join(text_list)
news["content"] = news["content"].apply(join)

news["tag"] = news["tag"].map({"详细全文":0, "国内":0, "国际":1})

X = news["content"]
Y = news["tag"]

#文本向量化
tf = TfidfVectorizer(min_df=2, ngram_range=(1,2))
X = tf.fit_transform(X)

#特征选择
selector = SelectKBest(f_classif,k=20000)
selector.fit(X, Y)
X=selector.transform(X).astype(np.float32)

#切分数据，构建训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

#多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(X_train,Y_train)
predicted = clf.predict(X_test)

#计算准确率
print('准确率为：', metrics.accuracy_score(Y_test, predicted))




