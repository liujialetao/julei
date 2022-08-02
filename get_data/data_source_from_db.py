#!/usr/python/bin/python3.6
# -*- coding: utf8 -*-
# :Autor: lichuanhu
# :Description: 获取DB数据
# :Date: 2020.2.14

# 获取call_log表中call_log_txt字段内容的sql语句
# task_id 为任务id
# return: 返回sql语句
import os,sys
# https://blog.csdn.net/qq1154479896/article/details/87557149
path = sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
import pymysql
import emi_nlp_config


def get_no_match_item(eid,script_id,time_start,time_end, limit):
    sql = "select id,content from aicall_no_match_item where eid = {0} and script_id = {1} and call_time between {2} and {3} limit {4};".format(str(eid),str(script_id),str(time_start),str(time_end),str(limit))
    return sql
def get_standard_question_by_eid_and_str(id,txt):
    sql = "SELECT standard_question FROM gansugonghang.question where similar_question like '%{0}%' and enterprise_uid = {1};".format(txt,str(id))
    return sql

def get_cluster_by_id(id):
    sql = "select id,name,content from cluster\
                where id = {0} limit 1000 ;".format(str(id))
    return sql
def get_cluster(enterprise_uid):
    sql = "select id,name,content from cluster\
                where enterprise_uid = {0} limit 1000 ;".format(str(enterprise_uid))
    return sql

def get_question_standard_and_similar_question(enterprise_uid):
    sql = "select id,standard_question,similar_question from question\
                where enterprise_uid = {0} limit 2000 ;".format(str(enterprise_uid))
    return sql


def get_call_log_txt_str(task_id):
    sql = "select calllog_txt from calllog\
                where task_id = {0} limit 1000 ;".format(str(task_id))
    return sql

# 获取call_log表中call_log_txt字段内容的sql语句
# return: 返回所有call_log_txt sql语句
def get_call_log_txt_all_str():
    sql = "select calllog_txt from calllog "
    return sql

# 获取call_log表中call_log_txt字段内容的sql语句
# return: 返回所有call_log_txt sql语句
def get_call_log_txt_by_intention_type_str():
    sql = "select calllog_txt from calllog where intention_type = 1 "
    return sql

def get_content_by_script_id():
    sql = "SELECT content FROM aicall_no_match_item where  script_id = 70; "
    return sql

def get_question_by_id(id):
    sql = "SELECT similar_question FROM gansugonghang.question where id = {0} ;".format(str(id))
    return sql

# 执行语句
# sql为需要执行的输入语句
# return: 返回执行结果
def execute_sql(sql):
    env_status = emi_nlp_config.get_env_status()
    if env_status == "0":        
        items = emi_nlp_config.get_db_info_dev()
    elif env_status == "1":
        items = emi_nlp_config.get_db_info_test()
    elif env_status == "2":  
        items = emi_nlp_config.get_db_info_product()
    else:
        print("db env is invalid value")
        return None
    results = None   
    db = pymysql.connect(host = items["host"],user = items["user"],passwd = items["passwd"],db = items["db"],charset="utf8")
 
    try:
        
        # 使用cursor()方法获取操作游标 
        cursor = db.cursor()
        # 执行SQL语句
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print ("Error: unable to fetch data, {0}".format(e))
    finally: 
        db.close()
     
   # calllog_txt = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"+calllog_txt
    return results

