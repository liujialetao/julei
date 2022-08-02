# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 14:49
# @Author  : liujia
# @File    : make_files.py
import pandas as pd
from extract_key_words import extract_key_words

def make_result(textvecot, vocabulary,all_sentences, y_pred, filename):
    '''

    :param all_sentences:
    :param y_pred:
    :param filename:
    :return:
    '''

    # 根据聚类算法的标签，将结果返回
    clusters = []
    # 计算聚类的个数
    n = len(set(y_pred))-1
    for i in range(n):
        Se = pd.Series(all_sentences)[y_pred==i].reset_index(drop=True)
        # 获取关键词
        key_words = extract_key_words(textvecot, vocabulary, Se.to_list())
        Se.name = '|'.join(key_words)
        clusters.append(Se)


    if -1 in list(set(y_pred)):
        Se = pd.Series(all_sentences)[y_pred == -1].reset_index(drop=True)
        # 获取关键词
        key_words = extract_key_words(textvecot, vocabulary, Se.to_list())
        Se.name = '离群点|'+'|'.join(key_words)
        clusters.append(Se)


    df = pd.concat(clusters,axis=1)
    df.to_excel(filename)
    return df
'''
aa = pd.Series([1,2,3], name='a')
bb = pd.Series([4,5,6], name='b')
cc = pd.Series([4,5,6], name='b')

pd.concat([aa,bb,cc], axis=1)
   a  b  b
0  1  4  4
1  2  5  5
2  3  6  6
'''