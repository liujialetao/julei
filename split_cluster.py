# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 10:09
# @Author  : liujia
# @File    : split_cluster.py
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from make_files import make_result
import os
import config
import pandas as pd
import numpy as np

class Split_Cluster():
    def __init__(self, sentences, sentences_vec):
        self.sentences_vec = sentences_vec
        self.sentences = sentences

    def kmeans(self, random_state=0, n_clusters=5):
        # https://blog.csdn.net/qq_48314528/article/details/119904631?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22119904631%22%2C%22source%22%3A%22weixin_44613381%22%7D&ctrtid=3dDEh
        X = self.sentences_vec
        cluster = KMeans(verbose=0, n_clusters=n_clusters, random_state=random_state).fit(X)
        y_pred = cluster.labels_
        return y_pred

    def dbscan(self, eps=7, min_samples=3):
        X = self.sentences_vec
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        y_pred = cluster.labels_
        return y_pred


        # if control==1:
        #     for eps in [2+i*0.5 for i in range(30)]:
        #         for min_samples in range(2, 10):
        #             cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        #             y_pred = cluster.labels_
        #
        #             #统计离群点
        #             print('eps:{}    min_samples:{}'.format(eps, min_samples))
        #             print('离群点有：', np.sum(pd.Series(y_pred) == -1))
        #             filename = 'dbscan2/eps_{}_sample_{}.csv'.format(eps, min_samples)
        #             df = make_result(self.sentences, y_pred, os.path.join(config.result_directory, filename))
        # else:
        #     cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        #     y_pred = cluster.labels_
        #
        #     # 统计离群点
        #     print('eps:{}    min_samples:{}'.format(eps, min_samples))
        #     print('离群点有：', np.sum(pd.Series(y_pred) == -1))
        #     filename = 'dbscan2/eps_{}_sample_{}.csv'.format(eps, min_samples)
        #     df = make_result(self.sentences, y_pred, os.path.join(config.result_directory, filename))
        #
        # return y_pred