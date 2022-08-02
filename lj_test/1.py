# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 16:16
# @Author  : liujia
# @File    : 1.py

for eps in [1 + i * 0.5 for i in range(30)]:
    for min_samples in range(2, 10):
        eps = str(eps)
        filename = 'dbscan2/eps_{}_sample_{}.csv'.format(eps, min_samples)