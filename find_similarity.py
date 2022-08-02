# -*- coding: utf-8 -*-
# @Time    : 2022/8/1 17:23
# @Author  : liujia
# @File    : find_similarity.py

import numpy as np
import pandas as pd

def find_similarity_cluster(confusion_value):
    '''
    #找到相似的聚类
    :param confusion_value:
    :return:
    '''
    similarity_pair = []
    matrix_value = confusion_value.values
    matrix_value_transpose = np.transpose(matrix_value)

    select_matrix = (matrix_value!=0)*(matrix_value_transpose!=0)

    n = select_matrix.shape[0]
    for i in range(n):
        select_matrix[i][i]=0

    for i in range(n):
        for j in range(n):
            if select_matrix[i][j]!=0:
                similarity_pair.append(i,j)
    return similarity_pair