import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm
import EstimatorDataOp
import json
import glob
import sys
import time
import pandas as pd
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
#os.environ["TF_XLA_FLAGS"]='--tf_xla_cpu_global_jit'
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.contrib import predictor 
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from sklearn.utils import shuffle
tf.logging.set_verbosity(tf.logging.INFO)

record_size = 512
record_dir = "/home/tiejianjie/news_rec/data/rank_record_data/*"
model_dir = "/home/tiejianjie/news_rec/data/model_data"

def model_training(record_list, model_dir):
    print('train_record_list:{}\tsize:{}'.format(len(record_list), len(record_list) * record_size))
    os.system(f"rm -rf {model_dir}")
    session_config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True # 自适应
    #session_config.graph_options.optimizer_options.global_jit_level =  config_pb2.OptimizerOptions.ON_1

    estimator_config = tf.estimator.RunConfig(
        session_config=session_config,
        log_step_count_steps=10,
        save_summary_steps=10 if sup < 200 else 500,
        model_dir=f'{model_dir}/model_summary',
        #train_distribute=tf.distribute.MirroredStrategy(devices=['/gpu:0','/gpu:1']),
        save_checkpoints_steps=len(record_list) * record_size // (params['batch_size'] * 2),
        keep_checkpoint_max=2022,
    )
    estimator_spec = tf.estimator.Estimator(
        model_fn=model_fn.model_fn,
        params=None,
	config=estimator_config
    )
    estimator_spec.train(
        input_fn=lambda : estimator_op.record_input_fn(
            record_list,
            **params
        ),
        steps=None if sup > 100000 else sup,
        hooks=[
            tf.estimator.ProfilerHook(
                save_steps=4000,
                output_dir=f"{model_dir}/model_profiler",
                show_memory=True,
            ),
        ]
    )
    for ckpt_path in tf.train.get_checkpoint_state(f"{model_dir}/model_summary").all_model_checkpoint_paths[1:]:
        estimator_spec.export_saved_model(
            f"{model_dir}/saved_model",
            estimator_op.infer_input_fn,
            checkpoint_path=ckpt_path
        )

if __name__ == '__main__':
    start = time.time()
    ##### pre define ####
    sup = int(sys.argv[1])
    model_name = sys.argv[2]
    estimator_op = EstimatorDataOp.EstimatorDataOp(record_size)
    sys.path.append(f"ModelZoo/{model_name}")
    import model_fn
    params = model_fn.train_config
    ##### model training #####
    train_record = [_ for _ in glob.glob(record_dir) if '29_' not in _]
    """
    tf.enable_eager_execution()
    data_iter = estimator_op.record_input_fn(
        shuffle(train_record),
        100,
        cpu_num=38,
        gpu_num=1
    ).make_one_shot_iterator()
    z = data_iter.get_next()
    #col_list = ["user_clk_cnt", "user_exp_cnt", "user_cate1_exp_cnt", "user_cate1_clk_cnt", "user_cate2_clk_cnt", "user_cate2_exp_cnt", "clk_user_time_since_last", "clk_user_time_since_first"]
    print(tf.log(1+z["user_cate1_exp_cnt"]))
    print(tf.log(1+z["user_cate1_clk_cnt"]))
    """
    model_training(
        record_list=shuffle(train_record),
        model_dir=f"{model_dir}/{model_name}",
    )
    end = time.time()
    print(f'------- nn_train.py cost:{round(end-start,5)}s------')
