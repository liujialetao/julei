# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 11:01
# @Author  : liujia
# @File    : get_sentence.py

from get_data import data_source_from_db

#get_no_match_cluster(eid,script_id,time_start,time_end):

def get_no_match_sentence(eid,script_id,time_start,time_end,limit):
    '''
    从数据库获取未匹配的句子
    '''
    sql = data_source_from_db.get_no_match_item(eid=eid,script_id=script_id,time_start=time_start,time_end=time_end,limit=limit)
    results = data_source_from_db.execute_sql(sql)

    return results
pass

if __name__ == '__main__':
    eid = 1132
    script_id = 108908
    time_start = 1656604800
    time_end = 1656604800
    time_end = 1658764799
    limit = 2000
    sql = data_source_from_db.get_no_match_item(eid=eid,script_id=script_id,time_start=time_start,time_end=time_end,limit=limit)
    results = data_source_from_db.execute_sql(sql)
    pass
