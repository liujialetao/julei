# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 16:12
# @Author  : liujia
# @File    : utils.py
import pandas as pd

def smoothing_sentence_frequency(frequence_sentence, min_count=1):
    '''
    :param frequence_sentence:
    :param min_count:语料至少出现的次数，小于min_count将被丢弃
    :return:
    '''
    import math
    def re_cal(n):
        if n==1:
            return 1
        elif n<100:
            return 2
        elif n<1000:
            return 3
        else:
            return 4
        # if n<=2:
        #     return n
        # else:
        #     #自定义的平滑方法
        #     n = int(math.log(n))+2
        #     return

    # 根据语料频次，过滤语料
    frequence_sentence = [(content, count) for content, count in list(frequence_sentence.items()) if count >= min_count]
    frequence_sentence = sorted(dict(frequence_sentence).items(), key=lambda x: x[1], reverse=True)

    # 平滑语料频次
    frequence_sentence2 = [(content, re_cal(count)) for content, count in frequence_sentence]
    frequence_sentence4 = sorted(dict(frequence_sentence2).items(), key=lambda x: x[1], reverse=True)

    # 讲语料按频次复制，存进列表返回
    corpus_list = []
    for content, count in frequence_sentence4:
        corpus_list += [content] * count

    return corpus_list






def huizong(dfs):
    '''

    :param dfs:
    :return:
    '''




    pass




