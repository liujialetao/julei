# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 17:48
# @Author  : liujia
# @File    : get_sentence_vec.py

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import config
import os
from get_contents import get_contents
from split_cluster import Split_Cluster
from make_files import make_result
import pandas as pd
import json
from utils import smoothing_sentence_frequency
from extract_key_words import make_textvecot
import time
from machine_learning import Classifier, make_train_val_data, make_confusion_matrix
from find_similarity import find_similarity_cluster
from merge_data import groupeby_data


def get_sentence_vec_from_bert(sentences):

    # 读取bert文件位置
    config_path = config.config_path
    checkpoint_path = config.checkpoint_path
    dict_path = config.dict_path

    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

    # 利用模型生成句向量
    sentences_vec = []
    i=1
    t1 = time.time()
    for sentence in sentences:
        token_ids, segment_ids = tokenizer.encode(sentence)
        vec = model.predict([np.array([token_ids]), np.array([segment_ids])])
        sentence_vec = vec[0][0].reshape(1,-1)
        sentences_vec.append(sentence_vec)
        if i%100==0:
            print('获取bert词向量{}个，用时{}秒'.format(i, time.time()-t1))
        i+=1
    sentences_vec = np.array(sentences_vec).reshape(-1,768)

    return sentences_vec

def reorganize_data(df, sentences):
    '''
    重新整理数据，整理出待分类的句子  已经分好类的句子
    :param df:
    :param sentences:
    :return:
    '''
    dic = {}
    classified_sentences = []
    for indexs in df.columns:
        column_data = df[indexs]
        classified_sentences_ = column_data.dropna().to_list()
        dic[column_data.name] = classified_sentences_
        classified_sentences.extend(classified_sentences_)

    not_classified_sentences = pd.Series(sentences)[~pd.Series(sentences).isin(classified_sentences)].to_list()
    return not_classified_sentences, classified_sentences, dic

# # 1 获取待分类句子
# frequence_sentence = get_contents(eid=1132,script_id=108908,time_start=1656604800,time_end=1658764799,limit=4000)
#
# # # 1_2 句子个数进行平滑和特殊处理
# sentences = smoothing_sentence_frequency(frequence_sentence, min_count=1)
# with open(os.path.join(config.sentence_data_directory,'bx_list_0.json'), 'w') as f_obj:
#     json.dump(sentences, f_obj)
#
#
#
# # 2 对句子生成句向量
# sentences_vec = get_sentence_vec_from_bert(sentences=sentences)
# np.save(os.path.join(config.sentence_data_directory,'my_array_0'), sentences_vec)

with open(os.path.join(config.sentence_data_directory,'bx_list_0.json'), 'r') as f_obj:
    sentences = json.load(f_obj)
sentences_vec = np.load(os.path.join(config.sentence_data_directory,'my_array_0.npy'))
dic_sentence = dict(zip(sentences, sentences_vec))

n = 4

# 3 聚类算法划分聚类
while(True):
    # cluster_result_save_path = os.path.join(config.cluster_directory, str(n) + 'epoch')
    # if not os.path.exists(cluster_result_save_path):
    #     os.mkdir(cluster_result_save_path)
    #
    # # 1 寻找聚类
    # # eps_list = []
    # # min_samples_list = []
    # eps_list = [6,7,8]#[2+i*0.5 for i in range(30)]
    # min_samples_list = [3,4,5,6,7]#[3,4]#list(range(1, 11))
    # dfs = []
    # # 根据所有句子，制作textvecot vocabulary
    # textvecot, vocabulary = make_textvecot(sentences)
    #
    # for eps in eps_list:
    #     for min_samples in min_samples_list:
    #         y_pred = Split_Cluster(sentences, sentences_vec).dbscan(eps=eps, min_samples=min_samples)
    #         #统计离群点
    #         outliers = np.sum(pd.Series(y_pred) == -1)
    #
    #         print('eps:{}    min_samples:{}'.format(eps, min_samples), end='\t')
    #         print('离群点有:{}个'.format(outliers))
    #         filename = 'eps_{}_sample_{}_outliers{}.xlsx'.format(eps, min_samples, outliers)
    #         df_ = make_result(textvecot, vocabulary, sentences, y_pred, os.path.join(cluster_result_save_path, filename))
    #         dfs.append(df_)


    # 手动归类
    new_df = pd.read_excel(os.path.join(config.cluster_directory, 'manul_define_clusters'+str(n)+'.xlsx'),index_col=0)
    new_df = new_df[new_df.columns[~new_df.columns.str.contains('named')]]
    # 重新整理数据，准备训练 或者 重新进行聚类
    not_classified_sentences, classified_sentences, classified_dic = reorganize_data(new_df, sentences)
    not_classified_sentences_vecs = [dic_sentence[sentences] for sentences in not_classified_sentences]

    sentences = not_classified_sentences
    sentences_vec = np.array(not_classified_sentences_vecs)

    # 手动选择是否要进行分类操作
    # fenlei = input('是否要进行分类操作，输入1或者0：')
    fenlei = '1'
    if fenlei=='1':
        train_sentences, train_labels, dic_num2label, dic_label2num = make_train_val_data(classified_dic)
        classifier = Classifier(dic_sentence, dic_num2label, train_sentences, train_labels)
        classifier.split_data(test_size=0.3, random_state=42)
        classifier.svm_fit_predict_val_sentence()

        #找出预测错误的句子
        val_sentences = pd.DataFrame(classifier.val_sentences)
        val_sentences['原来所属聚类'] = [dic_num2label[ele] for ele in classifier.val_label]
        val_sentences['预测所属聚类'] = classifier.predict_val_label_chinese

        misclassified_sentences = val_sentences[~(val_sentences['原来所属聚类']==val_sentences['预测所属聚类'])]
        # 按意图进行groupby
        misclassified_sentences = groupeby_data(misclassified_sentences,['原来所属聚类','预测所属聚类'])
        filename = os.path.join(config.cluster_directory, 'misclassified_sentences_{}.xlsx'.format(n))
        misclassified_sentences.to_excel(filename)



        #预测未聚类的句子
        pred_labels = classifier.predict(not_classified_sentences)
        filename = os.path.join(config.cluster_directory, 'not_classified_sentences_{}.xlsx'.format(n))
        df_not_classified_sentences = pd.DataFrame({'sentece':not_classified_sentences, 'label':pred_labels})
        df_not_classified_sentences = groupeby_data(df_not_classified_sentences, ['label'])
        df_not_classified_sentences.to_excel(filename)


        print(123)
        filename = os.path.join(config.cluster_directory, 'confusion_{}.xlsx'.format(n))
        df_confusion = make_confusion_matrix([dic_num2label[ele] for ele in classifier.val_label],classifier.predict_val_label_chinese, filename=filename)


        # 决策
        # 找到找回率不高的类别
        each_class_recall = df_confusion['recall'][:-2]
        low_recall_class = each_class_recall[(each_class_recall != 0) * (each_class_recall < 1)]

        # 决策，根据当前的混淆矩阵，找到混淆的类   出错率高的类别
        confusion_value = df_confusion[:-2].iloc[:, :-2]
        similarity_pair = find_similarity_cluster(confusion_value)

        #



        # 类别推荐
        print(1)








