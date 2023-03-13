import yaml
import json
import pyspark
from pyspark.sql.types import *
import time
import sys
sys.path.append('../')
import utils

feature_dir = "/home/test-t/news_rec/data/feature_data"

base_sql = f"""
    SELECT
        *,
        struct(doc_id, ctx_exposedTime, ctx_refreshTimes, watch_time) as `user_seq_struct`,
        struct(user_id, ctx_exposedTime, ctx_refreshTimes, watch_time) as `doc_seq_struct`
    FROM
        df
"""
useq_sql = """
    user_id,
    user_struct_to_str(array_sort(
        collect_list(user_seq_struct), 
            (l, r) -> case 
            when l.ctx_exposedTime < r.ctx_exposedTime then -1
            when l.ctx_exposedTime > r.ctx_exposedTime then 1 
            else 0 end
        )) as `user_seq`
"""

dseq_sql = """
    doc_id,
    doc_struct_to_str(array_sort(
        collect_list(doc_seq_struct), 
            (l, r) -> case 
            when l.ctx_exposedTime < r.ctx_exposedTime then -1
            when l.ctx_exposedTime > r.ctx_exposedTime then 1 
            else 0 end
        )) as `doc_seq`
"""
action_schema = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("doc_id", IntegerType(), False),
    StructField("ctx_exposedTime", DoubleType(), False),
    StructField("ctx_network", IntegerType(), False),
    StructField("ctx_refreshTimes", IntegerType(), False),
    StructField("ctx_exposedPos", IntegerType(), False),
    StructField("is_click", IntegerType(), False),
    StructField("watch_time", DoubleType(), False),
])


def user_struct_to_str(x):
    return '|'.join(['{}#{}#{}#{}'.format(_['doc_id'], _['ctx_exposedTime'], _['watch_time'], _['ctx_refreshTimes']) for _ in x])

def doc_struct_to_str(x):
    return '|'.join(['{}#{}#{}#{}'.format(_['user_id'], _['ctx_exposedTime'], _['watch_time'], _['ctx_refreshTimes']) for _ in x])
    

if __name__ == '__main__':
    start = time.time()
    #### pre define ####
    sup = min(int(sys.argv[1]), 2000000000)
    spark_sess = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_sess.sparkContext.setLogLevel("FATAL")
    #### register udf ####
    spark_sess.udf.register("user_struct_to_str", user_struct_to_str, StringType())
    spark_sess.udf.register("doc_struct_to_str", doc_struct_to_str, StringType())
    #### read data ####
    spark_sess.read.csv(f"{feature_dir}/processed_train_info.csv", schema=action_schema, sep='\t', header=None).createTempView("df")
    for act_type in ['clk', 'exp']:
        if act_type == 'exp':
            sql_list = "UNION ALL".join(([f"(SELECT {useq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} GROUP BY user_id)" for d in range(29, 24, -1)]))
        elif act_type == 'clk':
            sql_list = "UNION ALL".join(([f"(SELECT {useq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} and is_click == 1 GROUP BY user_id)" for d in range(29, 24, -1)]))
        else:
            sql_list = "UNION ALL".join(([f"(SELECT {useq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} and is_click == 0 GROUP BY user_id)" for d in range(29, 24, -1)]))

        spark_sess.sql(sql_list).write.partitionBy('day').csv(
            f"{feature_dir}/{act_type}_user_seq.csv", 
            mode='overwrite', 
            sep='\t', 
            header=True
        )

    for act_type in ['clk', 'exp']:
        if act_type == 'exp':
            sql_list = "UNION ALL".join(([f"(SELECT {dseq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} GROUP BY doc_id)" for d in range(29, 24, -1)]))
        elif act_type == 'clk':
            sql_list = "UNION ALL".join(([f"(SELECT {dseq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} and is_click == 1 GROUP BY doc_id)" for d in range(29, 24, -1)]))
        else:
            sql_list = "UNION ALL".join(([f"(SELECT {dseq_sql},{d} as `day` FROM ({base_sql}) WHERE day < {d} and is_click == 0 GROUP BY doc_id)" for d in range(29, 24, -1)]))

        spark_sess.sql(sql_list).write.partitionBy('day').csv(
            f"{feature_dir}/{act_type}_doc_seq.csv", 
            mode='overwrite', 
            sep='\t', 
            header=True
        )
