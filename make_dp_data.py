# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 15:23
# @Author  : liujia
# @File    : make_dp_data.py
import config
import numpy as np
import pandas as pd
import os
import json
from merge_data import groupeby_data
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def StratifiedKFold_sentences(sentences, labels, n_splits=3, shuffle=True):
    # 参考博客 https://blog.csdn.net/qq_41917697/article/details/112713507
    '''

    :param sentences: 
    :param label: 
    :return: 
    '''

    kfolds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    train_ls = []
    for train_index_ls,val_index_ls in kfolds.split(sentences,labels):
        train_sentences_ = np.array(sentences)[train_index_ls].tolist()
        train_lables_ = np.array(labels)[train_index_ls].tolist()
        val_sentences_ = np.array(sentences)[val_index_ls].tolist()
        val_lables_ = np.array(labels)[val_index_ls].tolist()
        train_ls.append([train_sentences_, train_lables_, val_sentences_,val_lables_])
    return train_ls


def make_dp_data(directory, dic_num2label, sentences, labels=None, data_type='train', predict_sentences=None):

    '''
    给transformers模型准备训练data
    :param epoch:
    :param dic_num2label:
    :param sentences:
    :param labels:
    :param data_type:
    :return:
    '''
    df = pd.DataFrame()
    if labels != None:
        df['label'] = [dic_num2label[ele] for ele in labels]
    df['text'] = sentences

    # 创建保存文件的文件夹
    filename = os.path.join(config.cluster_directory, directory)
    if not os.path.exists(filename):
        os.mkdir(filename)


    filename1 = os.path.join(filename, data_type) +'.csv'
    df.to_csv(filename1, index=0)
    return df



def make_transformer_label_list(epoch, dic_num2label):
    labels = [value for key,value in dic_num2label.items()]
    filename = os.path.join(config.cluster_directory, 'labels_{}.json'.format(epoch))

    with open(os.path.join(filename), 'w') as f_obj:
        json.dump(labels, f_obj)




def get_and_compare_predict_result(label_map, dic_num2label, p_directory, directory_name, original_filename, predict_filename):

    #所有数据所在的目录
    directory = os.path.join(p_directory, directory_name)
    # 读取原始数据
    df_orginal = pd.read_csv(os.path.join(directory, original_filename))
    df_orginal.rename(columns={'label': 'cluster_labels'}, inplace=True)
    #读取预测数据
    df_predict = pd.read_csv(os.path.join(directory, predict_filename), index_col=0)
    df_predict.rename(columns={'labels': 'predict_labels'}, inplace=True)
    df_predict['predict_labels'] = df_predict['predict_labels'].map(label_map)
    #合并数据
    df = pd.concat([df_orginal, df_predict], axis=1)

    if 'cluster_labels' in df.columns:
        df['预测结果是否相同'] = df['predict_labels'] == df['cluster_labels']

    if 'predict' in predict_filename:
        df.to_excel(os.path.join(directory, '待分类样本.xlsx'))
    else:
        df.to_excel(os.path.join(directory, '训练集和验证集样本.xlsx'))

    return df
