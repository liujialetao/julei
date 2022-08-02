# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 14:26
# @Author  : liujia
# @File    : config.py

import os

#存放原始语料
sentence_data_directory = 'data2_zhongan'

#存放聚类结果
result_directory = 'cluster_result/20220802'
cluster_directory = os.path.join(result_directory, 'extract_key_words')
if not os.path.exists(cluster_directory):
    os.makedirs(cluster_directory)


# bert模型相关参数
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'