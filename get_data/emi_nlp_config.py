#!/usr/python/bin/python3.6
# -*- coding: utf8 -*-
# :Autor: lichuanhu
# :Description: 获取DB数据
# :Date: 2020.2.14

# 获取call_log表中call_log_txt字段内容的sql语句
# task_id 为任务id
# return: 返回sql语句
import configparser
import os
file = 'get_data/emi_nlp.ini'

def get_eid():
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    sections = con.sections()
    items = con.items('eid')
    items = dict(items)
    return items["emi_list"]

def get_db_info_dev():
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    sections = con.sections()
    items = con.items('db_info_dev')
    items = dict(items)
    return items

def get_db_info_test():
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    sections = con.sections()
    items = con.items('db_info_test')
    items = dict(items)
    return items

def get_db_info_product():
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    sections = con.sections()
    items = con.items('db_info_product')
    items = dict(items)
    return items

def get_env_status():
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    sections = con.sections()
    items = con.items('env')
    items = dict(items)
    return items["env_status"]