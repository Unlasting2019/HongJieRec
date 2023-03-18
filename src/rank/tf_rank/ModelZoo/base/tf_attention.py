import tensorflow.compat.v1 as tf
import tf_layer
from tensorflow.contrib import layers as tcl
from tensorflow.keras.layers import LayerNormalization 
import math


def SASRecTransformer( 
    trm_inps, 
    trm_attri,
    attention_heads,
    mode,
    casual_mask
):
    def reshape_to_batch(x):
        x = tf.reshape(x, [-1, seq_len, head_num, head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, seq_len, head_dim])
        return x

    def reshape_from_batch(x):
        x = tf.reshape(x, [-1, head_num, seq_len, head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, seq_len, head_num * head_dim])
        return x

    def attention_op(qkv_inp, attr_logit, mask):
        q, k, v = qkv_inp
        attr_prob = tf.transpose(tf.layers.dense(attr_logit, 1, activation=tf.nn.sigmoid), [0, 2, 1])
        qk_logit = tf.einsum("bqd,bkd->bqk", q, k) / math.sqrt(head_dim) 
        qk_prob = tf.nn.softmax(qk_logit-1e9*mask, axis=-1) * attr_prob
        att_out = tf.einsum("bqk,bkd->bqd", qk_prob, v)
        return att_out

    seq_len = tf.shape(trm_inps)[1]
    head_dim, head_num = int(trm_inps.shape[-1]) // attention_heads, attention_heads

    with tf.variable_scope("MultiHeadAttention"):
        q = k = v = trm_inps
        batch_q = reshape_to_batch(tf.layers.dense(
            q, units=q.shape[-1], use_bias=False))
        batch_k = reshape_to_batch(tf.layers.dense(
            k, units=k.shape[-1], use_bias=False))
        batch_v = reshape_to_batch(tf.layers.dense(
            v, units=v.shape[-1], use_bias=False))
        batch_attr = reshape_to_batch(tf.layers.dense(
            tf.concat(trm_attri, -1), q.shape[-1], activation=tf.nn.leaky_relu
        ))
        batch_mha = reshape_from_batch(attention_op(
            [batch_q, batch_k, batch_v], batch_attr, casual_mask))
        mha_o = tf.layers.dense(batch_mha, units=q.shape[-1], use_bias=False) + q
        tf.summary.histogram("mha_q", batch_q)
        tf.summary.histogram("mha_k", batch_k)
        tf.summary.histogram("mha_v", batch_v)
        tf.summary.histogram("mha_batch", batch_mha)
        tf.summary.histogram("mha_o", mha_o)

    with tf.variable_scope("FeedForwardNetWork"):
        ffn1 = mha_o
        ffn2 = tf.layers.dense(ffn1, ffn1.shape[-1], activation=tf.nn.leaky_relu, use_bias=False)
        ffn3 = tf.layers.dense(ffn2, ffn1.shape[-1], use_bias=False)
        ffn4 = ffn1 + ffn3
        tf.summary.histogram("ffn1",ffn1)
        tf.summary.histogram("ffn2",ffn2)
        tf.summary.histogram("ffn3",ffn3)
        tf.summary.histogram("ffn4",ffn4)

    return ffn4

def TargetAttention(
    target_vec,
    seq_vec,
    seq_mask,
    mode
):
    q, k, v = tf.reshape(
        tf.tile(target_vec, [1, tf.shape(seq_vec)[1]]),
        tf.shape(seq_vec) 
    ), seq_vec, seq_vec 
    
    qk_logit1 = tf.concat([q, k, q*k, q-k],axis=-1)
    qk_logit2 = tf.layers.dense(qk_logit1, 40, activation=tf.nn.sigmoid)
    qk_logit3 = tf.layers.dense(qk_logit2, 20, activation=tf.nn.sigmoid)
    qk_prob = tf.layers.dense(qk_logit3, 1, activation=None)
    tf.summary.histogram("TargetAttention_qk_logit1", qk_logit1)
    tf.summary.histogram("TargetAttention_qk_logit2", qk_logit2)
    tf.summary.histogram("TargetAttention_qk_logit3", qk_logit3)
    tf.summary.histogram("TargetAttention_qk_prob", qk_prob)

    att_oup = tf.einsum("bk,bkd->bd", tf.squeeze(qk_prob,-1)*(1-seq_mask), v)
    att_oup = tf.keras.layers.LayerNormalization(name='target_att_out_ln')(att_oup)
    tf.summary.histogram("TargetAttention_oup", att_oup)
    return att_oup
