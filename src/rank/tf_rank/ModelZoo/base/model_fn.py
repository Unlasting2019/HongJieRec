import tensorflow.compat.v1 as tf
import math
import feature_operators as feature_op
from feature_operators import *
import sys
import tf_layer
from functools import reduce

model_config = {
    "sparse_reg":1e-6,
    "dense_reg":1e-6,
    "tower_mlp":{
        "mlp_cf":[
            [128,  lambda x, y : tf_layer.dice(x, y, "tower_mlp_dice1")],
            [1,    lambda x, y : tf.identity(x)]
        ],
        "pc":False,
    },
    'share_mlp':{
        "mlp_cf":[
            [512, lambda x, y : tf_layer.dice(x, y, "share_mlp_dice1")]
        ],
        "pc":True,
    },
    'ple_layer':{
        "layer_num":1,
        "tasks":["is_click","watch_time"],
        "task_dim":64,
        "expert_num":1,
        "expert_dim":256,
        "pc":False,
    }
}
train_config = {
    "batch_size": 16,
    "cpu_num": 38,
    "gpu_num": 2,
}
infer_config = {
    "batch_size": 12,
    "cpu_num": 38,
    "gpu_num": 2,
}
pc_schema = [
    {'DenseSingleOp':{'field':'user_id', 'emb_size':50, 'vocab_size':1029718, 'stop_gradient':True}},
    {'DenseSingleOp':{'field':'doc_id', 'emb_size':50, 'vocab_size':347466, 'stop_gradient':True}},
]

deep_schema = [
    #### user_schema
    {'DenseSingleOp':{'vocab_size':2823, 'field':'user_device', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':2, 'field':'user_os', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':280, 'field':'user_province', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':699, 'field':'user_city', 'emb_size':50}},
    {'DenseKeyValueOp':{'vocab_size':4, 'field':'user_age', 'emb_size':50}},
    {'DenseKeyValueOp':{'vocab_size':2, 'field':'user_gender', 'emb_size':50}},
    #### doc_schema
    {'DenseKeyValueOp':{'vocab_size':191020, 'field':'doc_title', 'norm':'log', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':38, 'field':'doc_cate1', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':193, 'field':'doc_cate2', 'emb_size':50}},
    {'DenseKeyValueOp':{'vocab_size':713391, 'field':'doc_keywords', 'norm':'log', 'emb_size':50}},
    #### ctx_schema
    {'DenseSingleOp':{'vocab_size':1029718, 'field':'user_id', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':347466, 'field':'doc_id', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':10, 'field':'ctx_network', 'emb_size':50}},
    {'DenseSingleOp':{'vocab_size':1000, 'field':'ctx_refreshTimes', 'emb_size':50}},
    #### SequenceOp ####
    {'DenseSequenceOp':{
        'seq_doc':{'field':'clk_doc_id', 'vocab_size':347466, 'emb_size':40, 'name':'clk_seq_doc_id'},
        'target_doc':{'field':'doc_id', 'vocab_size':347466, 'emb_size':40, 'name':'clk_target_doc_id'},
        'seq_attr':[
            {'field':'clk_doc_cate1','vocab_size':38,'emb_size':40, 'name':'clk_seq_doc_cate1'},
            {'field':'clk_doc_cate2','vocab_size':193,'emb_size':40, 'name':'clk_seq_doc_cate2'},
            {'field':'clk_refreshTimes','vocab_size':1000,'emb_size':40, 'name':'clk_seq_refreshTimes'},
        ],
        'unit_fcs':[
            'clk_watchTime',
            'clk_doc_cate1_cnt',
            'clk_doc_cate2_cnt',
        ]}
    },
    {'DenseNumericOp':{'field':'user_exp_cnt', 'norm':'log10', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_clk_cnt', 'norm':'log10', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_watch_sum', 'norm':'log10', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'user_watch_mean', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'clk_user_time_since_last', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_time_since_first', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_time_since_mean', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_exp_cnt', 'norm':'log', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_clk_cnt', 'norm':'log', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_exp_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_clk_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_watch_sum', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'user_cate1_watch_mean', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'user_cate1_watch_sum_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate1_watch_mean_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate1_time_since_last', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate1_time_since_first', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate1_time_since_mean', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_exp_cnt', 'norm':'log', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_clk_cnt', 'norm':'log', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_exp_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_clk_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_watch_sum', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'user_cate2_watch_mean', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'user_cate2_watch_sum_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'user_cate2_watch_mean_ratio', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate2_time_since_last', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate2_time_since_first', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_user_cate2_time_since_mean', 'norm':'BN', 'emb_size':4}},
    {'DenseMultiNumericOp':{'field':'user_title_exp_cnt', 'norm':'log', 'emb_size':4, 'reduce_std':True}},
    {'DenseMultiNumericOp':{'field':'user_title_clk_cnt', 'norm':'log', 'emb_size':4, 'reduce_std':True}},
    {'DenseMultiNumericOp':{'field':'user_title_exp_ratio', 'emb_size':4}},
    {'DenseMultiNumericOp':{'field':'user_title_clk_ratio', 'emb_size':4}},
    {'DenseMultiNumericOp':{'field':'user_title_watch_sum', 'emb_size':4, 'norm_value':1000., 'reduce_std':True}},
    {'DenseMultiNumericOp':{'field':'user_title_watch_mean', 'emb_size':4, 'norm_value':1000., 'reduce_std':True}},
    {'DenseMultiNumericOp':{'field':'user_title_watch_sum_ratio', 'emb_size':4}},
    {'DenseMultiNumericOp':{'field':'user_title_watch_mean_ratio', 'emb_size':4, 'reduce_std':True}},
    {'DenseNumericOp':{'field':'doc_exp_cnt', 'norm':'log10', 'emb_size':4}},
    {'DenseNumericOp':{'field':'doc_clk_cnt', 'norm':'log10', 'emb_size':4}},
    {'DenseNumericOp':{'field':'doc_watch_sum', 'norm':'log10', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'doc_watch_mean', 'norm':'log', 'emb_size':4, 'norm_value':1000.}},
    {'DenseNumericOp':{'field':'clk_doc_time_since_last', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_doc_time_since_first', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'clk_doc_time_since_mean', 'norm':'BN', 'emb_size':4}},
    {'DenseNumericOp':{'field':'doc_refreshTimes_exp_cnt','norm':'log10','emb_size':4}},
    {'DenseNumericOp':{'field':'doc_refreshTimes_clk_cnt','norm':'log10','emb_size':4}},
    {'DenseNumericOp':{'field':'doc_refreshTimes_exp_ratio','emb_size':4}},
    {'DenseNumericOp':{'field':'doc_refreshTimes_clk_ratio','emb_size':4}},
]

linear_schema = {
    "is_click":[
        {'SparseMultiSingleOp':{'field':'clk_doc_product', 'vocab_size':int(1e+7)}},
        #### doc ####
        {'SparseNumericOp':{'field':'doc_exp_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'doc_clk_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'doc_clk_mean'}},
        {'SparseNumericOp':{'field':'doc_clk_mode'}},
        {'SparseNumericOp':{'field':'doc_clk_median'}},
        {'SparseNumericOp':{'field':'doc_clk_var'}},
        {'SparseNumericOp':{'field':'doc_clk_skew'}},
        {'SparseNumericOp':{'field':'doc_clk_kurt'}},
        #### doc_refreshTimes ####
        {'SparseNumericOp':{'field':'doc_refreshTimes_exp_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_mean_ratio'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_ratio'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_exp_ratio'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_mean'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_mode'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_median'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_var'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_skew'}},
        {'SparseNumericOp':{'field':'doc_refreshTimes_clk_kurt'}},
        #### user_title ####
        {'SparseMultiNumericOp':{'field':'user_title_exp_cnt', 'norm':'log'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_cnt', 'norm':'log'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_mean_ratio'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_ratio'}},
        {'SparseMultiNumericOp':{'field':'user_title_exp_ratio'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_mean'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_mode'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_median'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_var'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_skew'}},
        {'SparseMultiNumericOp':{'field':'user_title_clk_kurt'}},
        #### user ####
        {'SparseNumericOp':{'field':'user_exp_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_clk_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_clk_mean'}},
        {'SparseNumericOp':{'field':'user_clk_mode'}},
        {'SparseNumericOp':{'field':'user_clk_median'}},
        {'SparseNumericOp':{'field':'user_clk_var'}},
        {'SparseNumericOp':{'field':'user_clk_skew'}},
        {'SparseNumericOp':{'field':'user_clk_kurt'}},
        {'SparseNumericOp':{'field':'clk_user_time_since_first', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_since_last', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_since_mean', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_since_mean_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_diff_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_diff_mean', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_time_diff_mean_std', 'norm':'log'}},
        #### user cate1 ####
        {'SparseSingleOp':{'field':'user_cate1_product', 'vocab_size':int(1e+7)}},
        {'SparseNumericOp':{'field':'user_cate1_exp_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_mean_ratio'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_ratio'}},
        {'SparseNumericOp':{'field':'user_cate1_exp_ratio'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_mean'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_mode'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_median'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_var'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_skew'}},
        {'SparseNumericOp':{'field':'user_cate1_clk_kurt'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_since_first', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_since_last','norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_since_mean', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_since_mean_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_diff_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_diff_mean','norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate1_time_diff_mean_std', 'norm':'log'}},
        #### user cate2 ####
        {'SparseSingleOp':{'field':'user_cate2_product', 'vocab_size':int(1e+7)}},
        {'SparseNumericOp':{'field':'user_cate2_exp_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_cnt', 'norm':'log'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_mean_ratio'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_ratio'}},
        {'SparseNumericOp':{'field':'user_cate2_exp_ratio'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_mean'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_mode'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_median'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_var'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_skew'}},
        {'SparseNumericOp':{'field':'user_cate2_clk_kurt'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_since_first', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_since_last', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_since_mean', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_since_mean_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_diff_std', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_diff_mean', 'norm':'log'}},
        {'SparseNumericOp':{'field':'clk_user_cate2_time_diff_mean_std', 'norm':'log'}},
    ],
    "watch_time":[
        #### user ####
        {"SparseNumericOp":{'field':'user_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'user_watch_mean', 'norm':'log', 'norm_value':1000.}},
        #### user_cate1
        {"SparseNumericOp":{'field':'user_cate1_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'user_cate1_watch_mean', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'user_cate1_watch_sum_ratio'}},
        {"SparseNumericOp":{'field':'user_cate1_watch_mean_ratio'}},
        #### user_cate2
        {"SparseNumericOp":{'field':'user_cate2_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'user_cate2_watch_mean', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'user_cate2_watch_sum_ratio'}},
        {"SparseNumericOp":{'field':'user_cate2_watch_mean_ratio'}},
        #### user_title
        {"SparseMultiNumericOp":{'field':'user_title_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseMultiNumericOp":{'field':'user_title_watch_mean', 'norm':'log', 'norm_value':1000.}},
        {"SparseMultiNumericOp":{'field':'user_title_watch_sum_ratio'}},
        {"SparseMultiNumericOp":{'field':'user_title_watch_mean_ratio'}},
        #### doc ####
        {"SparseNumericOp":{'field':'doc_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'doc_watch_mean', 'norm':'log', 'norm_value':1000.}},
        #### doc_refreshTimes
        {"SparseNumericOp":{'field':'doc_refreshTimes_watch_sum', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'doc_refreshTimes_watch_mean', 'norm':'log', 'norm_value':1000.}},
        {"SparseNumericOp":{'field':'doc_refreshTimes_watch_sum_ratio'}},
        {"SparseNumericOp":{'field':'doc_refreshTimes_watch_mean_ratio'}},
    ]
}

def get_train_ops(model_loss):
    update_ops = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    print('update_ops:',update_ops)
    """ deep embedding """
    deep_sparse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "embedding_part")
    deep_sparse_grads = tf.gradients(model_loss,  deep_sparse_vars)
    (deep_sparse_grads, _) = tf.clip_by_global_norm(deep_sparse_grads, 1)
    deep_sparse_ops = tf.train.AdamOptimizer(5e-3).apply_gradients(zip(deep_sparse_grads, deep_sparse_vars))
    for var, grad in zip(deep_sparse_vars, deep_sparse_grads):
        if grad is None: print(var)
        tf.summary.histogram(f"emb_grad/{var.name.replace(':','_')}", grad)
    """ deep mlp """
    deep_dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mlp_part")
    deep_dense_grads = tf.gradients(model_loss, deep_dense_vars)
    (deep_dense_grads, _) = tf.clip_by_global_norm(deep_dense_grads, 1)
    deep_dense_ops = tf.train.AdamOptimizer(1e-3).apply_gradients(zip(deep_dense_grads, deep_dense_vars))
    for var, grad in zip(deep_dense_vars, deep_dense_grads):
        if grad is None: print(var)
        tf.summary.histogram(f"deep_grad/{var.name.replace(':','_')}", grad)
    """ linear_part """
    linear_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "linear_part")
    linear_grads = tf.gradients(model_loss,  linear_vars)
    (linear_grads, _) = tf.clip_by_global_norm(linear_grads, 1)
    linear_ops = tf.train.FtrlOptimizer(1e-2).apply_gradients(zip(linear_grads, linear_vars))

    for var, grad in zip(linear_vars, linear_grads):
        if grad is None: print(var)
        tf.summary.histogram(f"linear_grad/{var.name.replace(':','_')}", grad)

    train_ops = tf.group(*[linear_ops, deep_sparse_ops, deep_dense_ops])
    with tf.control_dependencies([update_ops, train_ops]):
        return tf.assign_add(tf.train.get_global_step(), 1).op

def model_fn(features, labels, mode, params):
    print(f"parse_field:{len(features)}")
    print(f"deep_schema:{len(deep_schema)}")
    print(f"click_linear_schema:{len(linear_schema['is_click'])}")
    print(f"watch_linear_schema:{len(linear_schema['watch_time'])}")
    f_op = feature_op.FeatureOp(features, mode, model_config['sparse_reg'], model_config['dense_reg'])

    with tf.variable_scope("linear_part"):
        click_inputs = [f_op.lookup_fn(**conf) for conf in linear_schema["is_click"]]
        watch_inputs = [f_op.lookup_fn(**cf) for cf in linear_schema["watch_time"]]
        click_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer, name='click_bias')
        watch_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer, name='watch_bias')
        linear_click_logits = tf.add_n(click_inputs) + click_bias
        linear_watch_logits = tf.add_n(watch_inputs) + watch_bias 
        tf.summary.histogram("linear_watch_logits", linear_watch_logits)
        tf.summary.histogram("linear_click_logits", linear_click_logits)

    with tf.variable_scope("embedding_part", reuse=tf.AUTO_REUSE):
        f_emb = tf.concat([f_op.lookup_fn(**cf) for cf in deep_schema], -1)
        pc_emb = tf.concat([f_op.lookup_fn(**cf) for cf in pc_schema], -1)

    layer_op = tf_layer.LayerOp(features, mode, pc_emb)
    with tf.variable_scope("mlp_part"):
        with tf.variable_scope("shared_bottom_fc"):
            share_fc_oup = layer_op.mlp_layer(f_emb, **model_config["share_mlp"])
        with tf.variable_scope("ple_layer"):
            mtl_oup = layer_op.ple_layer(share_fc_oup, **model_config["ple_layer"])
        with tf.variable_scope("click_tower"):
            deep_click_logits = layer_op.mlp_layer(mtl_oup["is_click"], **model_config["tower_mlp"])
        with tf.variable_scope("watch_tower"):
            deep_watch_logits = layer_op.mlp_layer(mtl_oup["watch_time"],**model_config["tower_mlp"])

        click_logit = deep_click_logits + linear_click_logits
        watch_logit = deep_watch_logits + linear_watch_logits
        tf.summary.histogram("click_logit",click_logit)
        tf.summary.histogram("watch_logit:",watch_logit)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'is_click':tf.nn.sigmoid(click_logit)},
        )
    
    with tf.variable_scope('loss_part'):
        click_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(features['is_click'], tf.float32),
                logits=tf.reshape(click_logit, [-1])
            )
        )
        watch_mask = tf.cast(tf.reshape(features["watch_time"]>0., [-1]), tf.float32)
        watch_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(tf.cast(features["watch_time"] > 5, tf.float32), [-1]),
            logits=tf.reshape(watch_logit, [-1]),
        )  * watch_mask) / tf.reduce_sum(watch_mask)

        regularizer_loss = tf.losses.get_regularization_loss()
        model_loss = click_loss + watch_loss + regularizer_loss

        tf.summary.scalar('click_loss', click_loss)
        tf.summary.scalar('watch_loss', watch_loss)
        tf.summary.scalar('watch_mask', tf.reduce_sum(watch_mask))

        tf.summary.scalar("reg_loss", regularizer_loss)
        tf.summary.scalar('model_loss',model_loss)

    train_ops = get_train_ops(model_loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=model_loss,
        train_op=train_ops,
        training_hooks=[
            tf.estimator.LoggingTensorHook(
                tensors={
                    'click_loss':click_loss, 
                    'watch_loss':watch_loss, 
                    'reg_loss':regularizer_loss
                }, 
                every_n_iter=10,
            )
        ]
    )
