# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 10:48
# @Author  : liujia
# @File    : get_contents.py


from get_data.get_sentence import get_no_match_sentence
from collections import Counter


def get_contents(eid=1132,script_id=108908,time_start=1656604800,time_end=1658764799,limit=4000):
    '''

    :param eid:
    :param script_id:
    :param time_start:
    :param time_end:
    :param limit: SQL语句中的limit意义相同
    :return:
    '''

    #数据库获取未匹配句子
    results = get_no_match_sentence(eid=eid,script_id=script_id,time_start=time_start,time_end=time_end,limit=limit)
    # 找到高频的未匹配句子
    no_match_sentence = list(zip(*results))[1]
    frequence_sentence = Counter(no_match_sentence)
    return frequence_sentence

'''
从数据库获取no_match句子，去重排序后，获取待聚类的内容
:return:
'''
