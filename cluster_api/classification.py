# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 14:23
# @Author  : liujia
# @File    : classification.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from cluster_api.run_lj import start_train
import pandas as pd
import numpy as np
import os
import config
import json

app_classification = APIRouter()

class Data(BaseModel):
    train_args: dict = Field(default={'epochs': 3, 'lr': 2e-05})
    cluster_sentence: dict = Field(default={'我们~':["我们啊。", "我们。", "我们啊", "我们"],
                                            '没申请~':["我没申请啊。", "我没有申请啊。", "我没有申请。","我没申请啊", "我没有申请啊", "我没有申请"]},
                                   description="有标注的数据")
    sentence_to_be_predicted: List[str] = Field(default=['没申请呢','哦，我还没申请哦','我们啊'], description="待预测标签的数据")



@app_classification.post('/Classification')
def smoothing_frequency(
    data:Data
):
    train_args = data.train_args
    cluster_sentence = data.cluster_sentence
    sentence_to_be_predicted = data.sentence_to_be_predicted

    #处理数据
    df_train = pd.DataFrame([], columns=['label', 'text'])
    for key,value in cluster_sentence.items():
        maxtric_T = np.array([[key]*len(value), value]).transpose()
        df_ = pd.DataFrame(maxtric_T, columns=['label', 'text'])
        df_train = df_train.append(df_)
    df_train = df_train.append(pd.Series(['魑魅魍魉','魑魅魍魉'], index=['label','text']),  ignore_index=True)
    #将数据保存到训练需要的位置
    # 创建保存文件的文件夹
    df_train.to_csv(config.train_file, index=0)
    df_train.to_csv(config.dev_file, index=0)


    df_test = pd.DataFrame(sentence_to_be_predicted, columns=['text'])
    df_test.to_csv(config.test_file, index=0)
    #训练
    start_train(train_args)
    # 返回预测的数据
    df_result = pd.read_csv(os.path.join(config.save_directory, 'predict.csv'), index_col=0)
    with open(config.label_map, 'r') as f:
        label_map = json.load(f)
    id2label = dict([(value,key) for key,value in label_map.items()])
    df_result['labels'] = df_result['labels'].map(id2label)

    # 返回dic类型数据
    dic = {}
    dic['labels'] = df_result['labels'].to_list()
    dic['scroe'] = df_result['scroe'].tolist()
    return dic

