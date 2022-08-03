# -*- coding: utf-8 -*-
# @Time    : 2022/8/1 14:40
# @Author  : liujia
# @File    : machine_learning.py

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

class Classifier():
    '''多种分类器'''
    def __init__(self, dic_sentence, dic_num2label, sentences, lables):
        self.dic_sentence = dic_sentence
        self.dic_num2label = dic_num2label
        self.sentences = sentences
        self.labels = lables


    def split_data(self, test_size=0.3, random_state=42):
        self.train_sentences, self.val_sentences, self.train_label, self.val_label = train_test_split(self.sentences, self.labels, test_size=test_size, random_state=random_state)


    def gaussion_nb_fit_predict_val_sentence(self):
        self.classifier = GaussianNB()
        self.general_fit()
        self.predict_val_label_chinese = self.predict(self.val_sentences)

    def svm_fit_predict_val_sentence(self):
        self.classifier = SVC()
        self.general_fit()
        self.predict_val_label_chinese = self.predict(self.val_sentences)

    def general_fit(self):
        X_train = self.transpose_chinese2matrix(self.train_sentences)
        self.classifier.fit(X_train, self.train_label)


    def predict(self, x_predict_sentences):
        # 输入句子得到预测标签
        X_predict = np.array([ self.dic_sentence[ele] for ele in x_predict_sentences])
        y_pred = self.classifier.predict(X_predict)
        y_pred = self.get_chinese_label(y_pred)
        return y_pred


    def transpose_chinese2matrix(self, chinese_sentence_list):
        data_matrix = np.array([self.dic_sentence[ele] for ele in chinese_sentence_list])
        return data_matrix


    def get_chinese_label(self, y_number):
        chinese_label = [self.dic_num2label[ele] for ele in y_number]
        return chinese_label


def make_train_val_data(classified_dic):
    '''
    根据传入数据，制作训练数据集   兼顾机器学习和深度学习
    :param classified_dic:
    :return:
    '''
    label_list = [key for key, value in classified_dic.items()]
    dic_num2label = dict(zip(range(len(label_list)), label_list))
    dic_label2num = dict([val, key] for key, val in dic_num2label.items())

    # 制作训练data
    train_X = []
    train_Y = []
    for key, value in classified_dic.items():
        train_X.extend(value)
        train_Y.extend([dic_label2num[key]] * len(value))

    return train_X, train_Y, dic_num2label, dic_label2num


def make_confusion_matrix(true_label, predict_label, filename):
    #混淆矩阵
    all_labels = list(set(true_label + predict_label))

    confusion = confusion_matrix(true_label, predict_label, all_labels)
    df_confusion = pd.DataFrame(confusion, columns=['predict_' + str(label) for label in all_labels],
                                index=['true_' + str(label) for label in all_labels])

    ##混淆矩阵添加统计信息，计算召回率
    df_confusion = df_confusion.append(
        pd.Series(df_confusion.values.sum(axis=0), index=df_confusion.columns, name='sum'))  # 添加的Se的index跟原df的列索引必须要对齐
    df_confusion['sum'] = df_confusion.values.sum(axis=1)  # 行索引可以不设置
    df_confusion['recall'] = np.diagonal(df_confusion) / df_confusion['sum'].values
    # 计算precision
    total_precsion = np.array(np.sum(np.diagonal(df_confusion)[:-1]) / np.diagonal(df_confusion)[-1])
    precision_Eachclass = pd.Series(
        np.append(np.diagonal(df_confusion) / df_confusion.loc[['sum'], :].values[0][:-1], total_precsion),
        index=df_confusion.columns, name='precision')
    # 添加precision
    df_confusion = df_confusion.append(precision_Eachclass)  # 列索引必须要对齐
    # 保存文件
    if not os.path.exists(filename):
        if filename.endswith('.csv'):
            df_confusion.to_csv(filename)
        else:
            df_confusion.to_excel(filename)
    return df_confusion
