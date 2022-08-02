# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 13:51
# @Author  : liujia
# @File    : extract_key_words.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def remove_punctuation(all_sentences):
    #去除句子中的标点符号
    new_all_sentences = []
    for sentence in all_sentences:
        new_sentence = ''
        for ele in sentence:
            if ele not in ',.?，。？、':
                new_sentence += ele
        new_all_sentences.append(new_sentence)
    return new_all_sentences


def find_max_index(y, number=3):
    '''
    找到np.array中，最大的几个数，对应的索引
    :param y:
    :param number: 索引的个数
    :return:
    '''
    index_list = []
    for i in range(number):
        index = np.argmax(y)
        if np.max(y)!=0:
            index_list.append(index)
            y[0, index]=0 #y是二维矩阵
        else:
            return index_list
    return index_list


def extract_key_words(textvecot, vocabulary, corpus):

    corpus_o = corpus
    corpus = remove_punctuation(corpus)
    corpus = [jieba.lcut(sentence) for sentence in corpus]
    corpus = [' '.join(sentence) for sentence in corpus]

    x = textvecot.transform(corpus).todense()
    y = np.sum(x, axis=0)
    high_frequecy_word_index = find_max_index(y, number=5)

    if high_frequecy_word_index==[]:
        key_words = list(set(corpus_o))
    else:
        key_words = []
        for index in high_frequecy_word_index:
            key_words.append(vocabulary[index])
    return key_words



def make_textvecot(sentences):
    # 根据所有句子，制作textvecot vocabulary
    all_sentences = remove_punctuation(sentences)
    all_sentences = [jieba.lcut(sentence) for sentence in all_sentences]
    all_sentences = [' '.join(sentence) for sentence in all_sentences]
    gjz_count = TfidfVectorizer()
    textvecot = gjz_count.fit(all_sentences)
    vocabulary = {y: x for x, y in textvecot.vocabulary_.items()}
    return textvecot, vocabulary


if __name__ == '__main__':

    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    str1 = ' '.join(list(con1))
    str2 = ' '.join(list(con2))
    str3= ' '.join(list(con3))

    example1 = [str1, str2, str3]

    gjz_count = TfidfVectorizer()
    textvecot = gjz_count.fit_transform(example1) #得到文档向量化的矩阵内容

    print(textvecot.todense()) #通过todense方法获取向量化矩阵,数据结构为numpy.matrixlib.defmatrix.matrix
    print(gjz_count.vocabulary_)#查看矩阵每个列对应的分词，数据结构为字典
    type(gjz_count) #数据结构为sklearn.feature_extraction.text.CountVectorizer
    type(textvecot.todense()) #数据结构为numpy.matrixlib.defmatrix.matrix
    type(gjz_count.vocabulary_) #数据结构为字典
    #原理就是通过ID查询单词或通过单词查询ID