from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

#例子
example1 = ['我们 还 没有 到 家',
            '你们 什么 时候 回来',
            '他 想 吃 肉']
gjz_count = TfidfVectorizer()
textvecot = gjz_count.fit_transform(example1) #得到文档向量化的矩阵内容

textvecot.todense() #通过todense方法获取向量化矩阵,数据结构为numpy.matrixlib.defmatrix.matrix
gjz_count.vocabulary_ #查看矩阵每个列对应的分词，数据结构为字典
type(gjz_count) #数据结构为sklearn.feature_extraction.text.CountVectorizer
type(textvecot.todense()) #数据结构为numpy.matrixlib.defmatrix.matrix
type(gjz_count.vocabulary_) #数据结构为字典
#原理就是通过ID查询单词或通过单词查询ID