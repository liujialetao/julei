# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 14:22
# @Author  : liujia
# @File    : split_clusters.py

from fastapi import APIRouter
from pydantic import BaseModel,Field
from typing import List
from sklearn.cluster import DBSCAN,KMeans
# from cluster_api.public_utils.get_sentences_vec import Get_Sentences_Vec
import config
import numpy as np
import pandas as pd
# from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer
from transformers import BertTokenizer,BertModel
import time
import json

class Get_Sentences_Vec():
    def __init__(self, sentences):
        self.sentences = sentences


    # def get_sentence_vec_from_bert(self):
    #     # 读取bert文件位置
    #     config_path = config.config_path
    #     checkpoint_path = config.checkpoint_path
    #     dict_path = config.dict_path
    #
    #     tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    #     model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    #
    #     # 利用模型生成句向量
    #     sentences_vec = []
    #     i=1
    #     t1 = time.time()
    #     for sentence in self.sentences:
    #         token_ids, segment_ids = tokenizer.encode(sentence)
    #         vec = model.predict([np.array([token_ids]), np.array([segment_ids])])
    #         sentence_vec = vec[0][0].reshape(1,-1)
    #         sentences_vec.append(sentence_vec)
    #         if i%100==0:
    #             print('获取bert词向量{}个，用时{}秒'.format(i, time.time()-t1))
    #         i+=1
    #     sentences_vec = np.array(sentences_vec).reshape(-1,768)
    #     print('sentences_vec',sentences_vec)
    #     return sentences_vec

    def get_sentence_vec_from_bert_transformer(self):
        # 加载字典和分词工具
        token = BertTokenizer.from_pretrained('../../pretrained_models/bert/')
        pretrained = BertModel.from_pretrained('../../pretrained_models/bert/')

        data = token.batch_encode_plus(batch_text_or_text_pairs=self.sentences,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=128,
                                       return_tensors='pt',
                                       return_length=True)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']

        vecs = pretrained(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

        result_vecs = [vec[0].cpu().detach().numpy() for vec in vecs.last_hidden_state]#.cpu().detach().numpy()
        return result_vecs

class Split_Cluster():
    def __init__(self, sentences):
        self.sentences = sentences
        self.sentences_vec = Get_Sentences_Vec(self.sentences).get_sentence_vec_from_bert_transformer()

    # def kmeans(self, random_state=0, n_clusters=5):
    #     # https://blog.csdn.net/qq_48314528/article/details/119904631?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22119904631%22%2C%22source%22%3A%22weixin_44613381%22%7D&ctrtid=3dDEh
    #     X = self.sentences_vec
    #     cluster = KMeans(verbose=0, n_clusters=n_clusters, random_state=random_state).fit(X)
    #     y_pred = cluster.labels_
    #     return y_pred
    def dbscan_for(self, eps_list, min_sample_list):
        dic_list = []
        for eps in eps_list:
            for min_sample in min_sample_list:
                y_pred = self.dbscan_(eps, min_sample)
                dic_ = self.merge_data_by_label(y_pred)
                dic_['eps'] = eps
                dic_['min_sample'] = min_sample

                dic_list.append(dic_)

        return dic_list

    def merge_data_by_label(self,y_pred):

        '''

        :param all_sentences:
        :param y_pred:
        :param filename:
        :return:
        '''
        all_sentences = self.sentences
        # 根据聚类算法的标签，将结果返回
        clusters = []
        dic = {}
        # 计算聚类的个数
        n = len(set(y_pred)) - 1
        for i in range(n):
            Se = pd.Series(all_sentences)[y_pred == i].reset_index(drop=True)
            clusters.append(Se)
            dic[i] = Se.to_list()

        if -1 in list(set(y_pred)):
            Se = pd.Series(all_sentences)[y_pred == -1].reset_index(drop=True)
            clusters.append(Se)
            dic[-1] = Se.to_list()

        return dic

    def dbscan_(self, eps=7, min_samples=3):
        X = self.sentences_vec
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        y_pred = cluster.labels_
        return y_pred


app_split_clusters = APIRouter()

class Data(BaseModel):
    sentence_list: List = Field(default=["我们。", "我们的。","我们啊。", "我们的。", "我没申请啊。", "我没有申请啊。", "我没有申请。"],
                                description="聚类")
    eps_list: List[float] = Field(default=[7.0,8.0], description="DBSCAN算法参数，邻域半径")
    min_sample_list: List[int] = Field(default=[2,3,4], description="DBSCAN算法参数，邻域附近点个数")

@app_split_clusters.post('/SplitClusters')
def split_clusters(data:Data, return_json=1):
    sentence_list = data.sentence_list
    eps_list = data.eps_list
    min_sample_list = data.min_sample_list

    m = Split_Cluster(sentence_list)
    result = m.dbscan_for(eps_list, min_sample_list)
    if return_json==1 or return_json=='1':
        result = json.dumps(result)
    return result


if __name__ == '__main__':
    m = Split_Cluster(['哦','好的'])
    print(1)






#
# app_smoothing_frequency = APIRouter()
#
#
# class Data(BaseModel):
#     sentence_list: List = Field(default=["好的好的。", "好的好的。", "我没申请啊。"], description="对词频进行平滑，例如 好的好的。出现500次，经过平滑处理后，保留4个。")#alias="所有句待聚类/分类的句子",
#     min_count: int = Field(default=1, description="小于min_count的文本，忽略不计")#alias="最少出现的次数",
#
# @app_smoothing_frequency.post('/SmoothingFrequency')
# def smoothing_frequency(
#     data:Data
# ):
#     sentence_list = data.sentence_list
#     return sentence_list
#
