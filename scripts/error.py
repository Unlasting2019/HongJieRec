import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark
import time
import sys
import pyspark
from pyspark.sql.types import *
import time
import sys

feature_dir = "/home/tiejianjie/news_rec/data/feature_data"

df = pd.read_csv("/home/tiejianjie/news_rec/data/feature_data/processed_train_info.csv/day=29/data.csv", nrows=int(sys.argv[1]), sep='\t')
df.columns = ['user_id', 'doc_id', 'ctx_exposedTime', 'ctx_network', 'ctx_refreshTimes', 'ctx_exposedPos', 'is_click', 'watchTime']
df['hour'] = df['ctx_exposedTime'].apply(lambda x : time.localtime(x/1000).tm_hour)

user_df = pd.read_csv("/home/tiejianjie/news_rec/data/feature_data/processed_user_info.csv", nrows=int(sys.argv[1]) * 10, sep='\t')
user_df.columns = ['user_id', 'user_device', 'user_os', 'user_province', 'user_city', 'user_age', 'user_gender']

doc_df = pd.read_csv("/home/tiejianjie/news_rec/data/feature_data/processed_doc_info.csv", nrows=int(sys.argv[1]) * 10, sep='\t')
doc_df.columns = ['doc_id', 'doc_title', 'doc_postTime', 'doc_picNum', 'doc_cate1', 'doc_cate2', 'doc_keywords']


y_pred = pd.read_csv("/home/tiejianjie/news_rec/src/rank/tf_rank/tmp/WideTrm1.csv", nrows=int(sys.argv[1])).rename(columns={'is_click':'pctr'})

df = df.merge(user_df, how='left', on='user_id').merge(doc_df, how='left', on='doc_id')
df['pctr'] = y_pred['pctr']

print('doc error')
#### single_col error 
for merge_col in [['doc_id'], ['doc_id','ctx_refreshTimes'], ['doc_id','hour'], ['doc_id', 'user_device'],['doc_id','user_os'],['doc_id','user_province'],['doc_id','user_city']]:
    pred_ = df.groupby(merge_col).agg({'pctr':['mean','count']})
    true_ = df.groupby(merge_col).agg({'is_click':['mean','count']})
    show_num = 50
    df_ = pred_.merge(true_, how='left', on=merge_col)
    df_['diff'] = df_[('is_click', 'mean')] - df_[('pctr', 'mean')]
    print('{} error rate: {}'.format(merge_col, df_[abs(df_['diff']) > 0.1].shape[0] / df_.shape[0]))

print('user error')
for merge_col in [['user_id'], ['user_id','ctx_refreshTimes'], ['user_id','hour'], ['user_id', 'doc_cate1'],['user_id','doc_cate2'],['user_id','doc_picNum']]:
    pred_ = df.groupby(merge_col).agg({'pctr':['mean','count']})
    true_ = df.groupby(merge_col).agg({'is_click':['mean','count']})
    show_num = 50
    df_ = pred_.merge(true_, how='left', on=merge_col)
    df_['diff'] = df_[('is_click', 'mean')] - df_[('pctr', 'mean')]
    print('{} error rate: {}'.format(merge_col, df_[abs(df_['diff']) > 0.1].shape[0] / df_.shape[0]))


