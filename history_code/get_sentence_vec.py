# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 17:48
# @Author  : liujia
# @File    : get_sentence_vec.py

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_sentence_vec1(sentences):

    # 根据sentences，得到句子向量
    config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

    sentences_vec = []
    for sentence in sentences:
        token_ids, segment_ids = tokenizer.encode(sentence)
        vec = model.predict([np.array([token_ids]), np.array([segment_ids])])
        sentence_vec = vec[0][0].reshape(1,-1)
        sentences_vec.append(sentence_vec)

    sentences_vec = np.array(sentences_vec).reshape(-1,768)

    return  sentences_vec


if __name__ == '__main__':
    get_sentence_vec1(['贷款利率多少', '利息怎么算的'])

# similarity_matric = cosine_similarity(x, x)
# df = pd.DataFrame(similarity_matric, columns=sentences, index=sentences)
# df.to_csv('词向量效果测试.csv')
pass
# def cal_similarity(sentence1, sentence2):
#     # token_ids, segment_ids = tokenizer.encode(u'北京最近的疫情情况得到了有效的控制')
#
#     token_ids, segment_ids = tokenizer.encode(sentence)
#     vec1 = model.predict([np.array([token_ids]), np.array([segment_ids])])
#
#     token_ids, segment_ids = tokenizer.encode(sentence2)
#     vec2 = model.predict([np.array([token_ids]), np.array([segment_ids])])
#
#
#     similarity = cosine_similarity(vec1[0][0].reshape(1,-1), vec2[0][0].reshape(1,-1))
#
#     return similarity[0][0]
#
# sentence1 = '我们'
# sentence2 = '我们的'
# similarity = cal_similarity(sentence1, sentence2)
# print(similarity)
