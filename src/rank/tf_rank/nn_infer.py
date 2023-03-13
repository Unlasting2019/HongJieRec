import pickle
import numpy as np
import re
from tqdm import tqdm
import EstimatorDataOp
import json
import glob
import yaml
import sys
import time
import pandas as pd
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["TF_XLA_FLAGS"]='--tf_xla_cpu_global_jit'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.enable_eager_execution()
from nn_train import *
from tensorflow.contrib import predictor 
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from sklearn.utils import shuffle


def model_predict(record_list, model_dir, size):
    print('infer_record_list:{}\tsize:{}\t{}'.format(len(record_list), len(record_list) * 512, record_list[:10]))
    for i, timestamp in enumerate(sorted(os.listdir(model_dir))):
        predictor_ = predictor.from_saved_model(f"{model_dir}/{timestamp}")
        data_iter = tf.data.TFRecordDataset(record_list, num_parallel_reads=params['cpu_num']).batch(params['batch_size']).prefetch(params['gpu_num'])
        y_pred = [predictor_({'tf_exp':tf_exp.numpy()})['is_click'] for tf_exp in tqdm(data_iter, total=size)]
        pd.DataFrame({
            "is_click":np.concatenate(y_pred, 0).reshape(-1).astype(np.float64)
        }).to_csv(f"tmp/{model_name}.csv",index=False)

if __name__ == '__main__':
    start = time.time()
    ##### pre define ####
    sup = int(sys.argv[1])
    model_name = sys.argv[2]
    estimator_op = EstimatorDataOp.EstimatorDataOp(record_size)
    sys.path.append(f"ModelZoo/{model_name}")
    import model_fn
    params = model_fn.infer_config
    ##### model training #####
    infer_record = sorted(
        [_ for _ in glob.glob(record_dir) if '29_' in _], 
        key=lambda x : int(re.split('/|_', x)[-1])
    )[:sup*10]
    model_predict(
        record_list=infer_record,
        model_dir=f"{model_dir}/{model_name}/saved_model",
        size=len(infer_record) // params['batch_size'],
    )
    end = time.time()
    print(f'------- nn_infer.py cost:{round(end-start,5)}s------')
