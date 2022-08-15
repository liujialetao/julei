# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 9:44
# @Author  : liujia
# @File    : run.py

import uvicorn
from fastapi import FastAPI
from cluster_api import app_smoothing_frequency, app_split_clusters, app_classification, app_extract_key_words

app = FastAPI(
    title='提供聚类划分功能',
    description='提供3个接口',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redocs',
)
app.include_router(app_smoothing_frequency, prefix='/Cluster', tags=['文本词频平滑处理'])
app.include_router(app_split_clusters, prefix='/Cluster', tags=['文本聚类'])
app.include_router(app_classification, prefix='/Cluster', tags=['文本分类'])
app.include_router(app_extract_key_words, prefix='/Cluster', tags=['聚类关键词提取'])

if __name__ == '__main__':
    uvicorn.run('run:app', host='0.0.0.0', port=8001, reload=True, debug=True, workers=1)









