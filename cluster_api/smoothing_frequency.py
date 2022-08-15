# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 14:17
# @Author  : liujia
# @File    : smoothing_frequency.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from collections import Counter
import json
import os
import config

app_smoothing_frequency = APIRouter()

def smoothing_sentence_frequency(frequence_sentence, min_count=1):
    '''
    :param frequence_sentence:
    :param min_count:语料至少出现的次数，小于min_count将被丢弃
    :return:
    '''
    import math
    def re_cal(n):
        print('???????')
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
    frequence_sentence = Counter(frequence_sentence)
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

class Data(BaseModel):
    sentence_list: List = Field(default=["好的好的。", "好的好的。", "我没申请啊。"], description="对词频进行平滑，例如 好的好的。出现500次，经过平滑处理后，保留4个。")#alias="所有句待聚类/分类的句子",
    min_count: int = Field(default=1, description="小于min_count的文本，忽略不计")#alias="最少出现的次数",

@app_smoothing_frequency.post('/SmoothingFrequency')
def smoothing_frequency(
    data:Data
):
    sentence_list = smoothing_sentence_frequency(data.sentence_list, data.min_count)
    return sentence_list

if __name__ == '__main__':
    with open(os.path.join(config.sentence_data_directory, 'bx_list_0.json'), 'r') as f_obj:
        all_sentences = json.load(f_obj)
    sentences = smoothing_sentence_frequency(all_sentences, 1)
