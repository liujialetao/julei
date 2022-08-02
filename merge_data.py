# -*- coding: utf-8 -*-
# @Time    : 2022/8/1 19:56
# @Author  : liujia
# @File    : merge_data.py

import pandas as pd

def groupeby_data(df, goupeby_list):

    for lable in goupeby_list:
        dfs = []
        for name, df_ in list(df.groupby([lable])):
            dfs.append(df_)
        df = pd.concat(dfs)

    return df
