# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 14:26
# @Author  : liujia
# @File    : config.py

import os

#存放原始语料
sentence_data_directory = 'sentence_data_directory'
if not os.path.exists(sentence_data_directory):
    os.makedirs(sentence_data_directory)

#存放聚类结果
result_directory = '20220812'
cluster_directory = os.path.join(result_directory, 'extract_key_words')
if not os.path.exists(cluster_directory):
    os.makedirs(cluster_directory)


# bert模型相关参数
bert_directory = 'chinese_L-12_H-768_A-12/'

config_path = os.path.join(bert_directory, 'bert_config.json')
print(os.path.abspath(config_path))
checkpoint_path = os.path.join(bert_directory, 'bert_model.ckpt')
dict_path =  os.path.join(bert_directory, 'vocab.txt')

# bert训练数据，保存数据
train_directory = os.path.join('cluster_api/dataset/fastapi/')
if not os.path.exists(train_directory):
    os.makedirs(train_directory)

train_file = os.path.join(train_directory, 'train') + '.csv'
dev_file = os.path.join(train_directory, 'dev') + '.csv'
test_file = os.path.join(train_directory, 'test') + '.csv'

save_directory = os.path.join('cluster_api/results/fastapi/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
label_map = os.path.join(save_directory,'label_map.json')