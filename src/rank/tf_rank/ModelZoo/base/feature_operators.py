import tensorflow.compat.v1 as tf
import tf_rnn
import math
from tensorflow.python.ops import string_ops, math_ops
import tf_attention
from tensorflow.contrib import layers as tcl

color_print = "\033[7m{}\033[0m"
emb_init = lambda x : tf.truncated_normal_initializer(0, 1 / math.sqrt(x))


class FeatureOp:
    def __init__(self, features, mode, sparse_reg, dense_reg):
        self.features = features
        self.mode = mode
        self.sparse_reg = sparse_reg
        self.dense_reg = dense_reg

    def lookup_fn(
        self,
        SparseSingleOp=None,
        SparseNumericOp=None,
        SparseMultiSingleOp=None,
        SparseMultiNumericOp=None,
        DenseSingleOp=None,
        DenseNumericOp=None,
        DenseMultiNumericOp=None,
        DenseKeyValueOp=None,
        DenseSequenceOp=None,
    ):
        if DenseSingleOp:
            return self.DenseSingleOp(**DenseSingleOp)
        elif DenseNumericOp:
            return self.DenseNumericOp(**DenseNumericOp)
        elif DenseKeyValueOp:
            return self.DenseKeyValueOp(**DenseKeyValueOp)
        elif DenseSequenceOp:
            return self.DenseSequenceOp(**DenseSequenceOp)
        elif DenseMultiNumericOp:
            return self.DenseMultiNumericOp(**DenseMultiNumericOp)
        elif SparseSingleOp:
            return self.SparseSingleOp(**SparseSingleOp)
        elif SparseNumericOp:
            return self.SparseNumericOp(**SparseNumericOp)
        elif SparseMultiSingleOp:
            return self.SparseMultiSingleOp(**SparseMultiSingleOp)
        elif SparseMultiNumericOp:
            return self.SparseMultiNumericOp(**SparseMultiNumericOp)
        else:
            raise ValueError("None Op Found")

    def SparseMultiSingleOp(self, field, vocab_size):
        print(color_print.format(f"{field} - SparseMultiSingleOp"))
        sp_ids = self.features[field].values
        segment_ids = self.features[field].indices[:, 0]
        sp_weight = tf.get_variable(
            shape=[vocab_size, 1],
            initializer=tf.zeros_initializer,
            name=f'SparseEmb_{field}',
            regularizer=tcl.l2_regularizer(self.sparse_reg)
        )
        segment_emb = tf.gather(sp_weight, sp_ids % vocab_size)
        x_vec = tf.math.segment_sum(segment_emb, segment_ids)

        tf.summary.scalar(f"{field}_vec", tf.reduce_mean(x_vec))
        print(f'- x_vec:{x_vec}')
        print(f'- x_emb:{sp_weight}')
        return x_vec

    def SparseMultiNumericOp(self, field, norm=None, norm_value=1.):
        print(color_print.format(f"{field} - SparseMultiNumericOp"))
        sp_val = tf.expand_dims(self.features[field].values, -1) / norm_value
        segment_ids = self.features[field].indices[:, 0]
        if norm == 'abs':
            norm_x = tf.sign(sp_val) * tf.log(1+tf.abs(sp_val))
        elif norm == 'log':
            norm_x = tf.log(1+sp_val)
        else:
            norm_x = sp_val

        x_weight = tf.get_variable(
            shape=[1],
            initializer=tf.zeros_initializer,
            regularizer=tcl.l2_regularizer(self.sparse_reg),
            name=f"SparseEmb_{field}"
        )
        x_vec = tf.math.segment_sum(norm_x * x_weight,segment_ids)


        tf.summary.histogram(f"{field}_emb", x_weight)
        tf.summary.scalar(f"{field}_vec", tf.reduce_mean(x_vec))
        print(f'- SparseMultiNumericOp:{x_vec}')
        return x_vec

    

    def SparseSingleOp(self, field, vocab_size):
        print(color_print.format(f"{field} - SparseSingleOp"))
        x_weight = tf.get_variable(
            shape=[vocab_size, 1],
            initializer=tf.zeros_initializer,
            regularizer=tcl.l2_regularizer(self.sparse_reg),
            name=f"SparseEmb_{field}"
        )
        x_vec = tf.gather(x_weight, self.features[field])

        tf.summary.scalar("{field}_vec", tf.reduce_mean(x_vec))
        print(f'- x_vec:{x_vec}')
        print(f'- x_emb:{x_weight}')
        return x_vec

    def SparseNumericOp(self, field, norm=None, norm_value=1.):
        print(color_print.format(f"{field} - SparseNumericOp"))
        x = self.features[field] / norm_value
        if norm == 'abs':
            norm_x = tf.sign(x) * tf.log(1+tf.abs(x))
        elif norm == 'log':
            norm_x = tf.log(1+x)
        else:
            norm_x = x

        x_weight = tf.get_variable(
            shape=[1],
            initializer=tf.zeros_initializer,
            regularizer=tcl.l2_regularizer(self.sparse_reg),
            name=f"SparseEmb_{field}"
        )
        x_vec = norm_x * x_weight

        tf.summary.histogram(f"{field}_emb", x_weight)
        tf.summary.scalar(f"{field}_vec", tf.reduce_mean(x_vec))
        print(f'- SparseNumericOp:{x_vec}')
        return x_vec


    def DenseMultiNumericOp(self, field, emb_size, name=None, norm=None, norm_value=1., reduce_std=False):
        print(color_print.format(f"{field} - DenseMultiNumericOp"))
        sp_val = tf.expand_dims(self.features[field].values, -1) / norm_value
        segment_ids = self.features[field].indices[:, 0]
        name = field if name is None else name
        if norm == 'abs':
            norm_x = tf.sign(sp_val) * tf.log(1+tf.abs(sp_val))
        elif norm == 'log':
            norm_x = tf.log(1+sp_val)
        elif norm == 'log10':
            norm_x = tf.log(1+sp_val) / tf.log(10.)
        elif norm == 'BN':
            norm_x = tf.layers.batch_normalization(
                sp_val,
                training = self.mode == tf.estimator.ModeKeys.TRAIN,
                name=f"DenseMultiNumericOp_{field}_BN",
            )
        else:
            norm_x = sp_val

        x_weight = tf.get_variable(
            shape=[emb_size],
            initializer=emb_init(emb_size),
            regularizer=tcl.l2_regularizer(self.dense_reg),
            name=f"{name}_emb",
        )
        x_vec = tf.math.segment_sum(norm_x * x_weight, segment_ids)
        if reduce_std:
            x_vec /= (tf.math.reduce_std(x_vec, -1, True)+1e-12)

        tf.summary.histogram(f"{name}_emb", x_weight)
        tf.summary.histogram(f"{name}_vec", x_vec)
        print(f'- DenseMultiNumericOp:{x_vec}')
        return x_vec

    def DenseNumericOp(self, field, emb_size, name=None, norm=None, norm_value=1.):
        print(color_print.format(f"{field} - DenseNumericOp"))
        x = self.features[field] / norm_value
        name = f"{field}" if name is None else name
        if norm == 'log':
            norm_x = tf.log(1+x)
        elif norm == 'log10':
            norm_x = tf.log(1+x) / tf.log(10.)
        elif norm == 'BN':
            norm_x = tf.layers.batch_normalization(
                x,
                training = self.mode == tf.estimator.ModeKeys.TRAIN,
                name=f"DenseNumeric_{field}_BN",
            )
        else:
            norm_x = x;

        x_weight = tf.get_variable(
            shape=[emb_size],
            initializer=emb_init(emb_size),
            regularizer=tcl.l2_regularizer(self.dense_reg),
            name=f"{name}_emb",
        )
        x_vec = norm_x * x_weight

        tf.summary.histogram(f"{name}_emb", x_weight)
        tf.summary.histogram(f"{name}_vec", x_vec)
        print(f'- DenseNumeric:{x_vec}')
        return x_vec

    def DenseSingleOp(self, vocab_size, field, emb_size, stop_gradient=False):
        print(color_print.format(f"{field} - DenseSingleOp"))
        x_weight = tf.get_variable(
            shape=[vocab_size+2, emb_size],
            name=field,
            initializer=emb_init(emb_size),
            regularizer=tcl.l2_regularizer(self.dense_reg),
        )
        x_vec = tf.gather(x_weight, self.features[field])

        tf.summary.histogram(f"{field}_vec", x_vec)
        print(f'- x_vec:{x_vec}')
        print(f'- x_emb:{x_weight}')
        return tf.stop_gradient(x_vec) if stop_gradient else x_vec

    def DenseKeyValueOp(self, vocab_size, field, emb_size, norm=None):
        print(color_print.format(f"{field} - DenseKeyValueOp"))
        sp_key = self.features[f"{field}_key"].values
        sp_val = self.features[f"{field}_val"].values
        segment_ids = self.features[f"{field}_key"].indices[:, 0]

        if norm == 'log':
            sp_val = tf.log(1+sp_val)
        elif norm == 'log10':
            sp_val = tf.log(1+sp_val) / tf.log(10.)

        sp_key_emb = tf.get_variable(
            shape=[vocab_size+2, emb_size],
            name=f"{field}_k",
            initializer=emb_init(emb_size),
            regularizer=tcl.l2_regularizer(self.dense_reg),
        )
        sp_val_emb = tf.get_variable(
            shape=[emb_size],
            name=f"{field}_v",
            initializer=emb_init(emb_size),
            regularizer=tcl.l2_regularizer(self.dense_reg)
        )
        sp_key_vec = tf.gather(sp_key_emb, sp_key)
        sp_val_vec = tf.einsum("b,d->bd", sp_val, sp_val_emb)
        sp_kv_vec = tf.math.segment_sum(sp_key_vec * sp_val_vec, segment_ids)

        print(f'- sp_k_emb:{sp_key_emb}')
        print(f'- sp_v_emb:{sp_val_emb}')
        print(f'- sp_kv_vec:{sp_kv_vec}')
        tf.summary.histogram(f"{field}_k_vec", sp_key_vec)
        tf.summary.histogram(f"{field}_v_vec", sp_val_vec)
        return sp_kv_vec

    def DenseSequenceOp(self, seq_doc, target_doc, seq_attr, unit_fcs):
        def seq_single(field, vocab_size, emb_size, name):
            x_emb = tf.get_variable(
                shape=[vocab_size+2, emb_size],
                name=name,
                initializer=emb_init(emb_size),
                regularizer=tcl.l2_regularizer(self.dense_reg),
            )
            x_vec = tf.gather(x_emb, self.features[field])
            print(f'- {field}_emb', x_emb)
            print(f'- {field}_vec', x_vec)

            tf.summary.histogram(f"Sequence_{field}_vec",x_vec)
            return x_vec

        def InterestUnit(fcs, emb_size):
            norm_x = tf.stack([tf.log(self.features[f]+1) for f in fcs], -1)
            norm_x_vec = tf.layers.dense(
                norm_x,
                units=emb_size,
                activation=tf.nn.sigmoid,
                kernel_initializer=emb_init(emb_size),
            )  * 2
            return norm_x_vec


        print(color_print.format(f"SequenceOp"))
        seq_vec = seq_single(**seq_doc)
        target_vec = seq_single(**target_doc)
        seq_attri = [seq_single(**fc) for fc in seq_attr]
        tf.summary.histogram("Sequence_seq_vec", seq_vec)
        tf.summary.histogram("Sequence_target_vec", target_vec)
        print(f'- seq_vec:{seq_vec}')
        print(f'- target_vec:{target_vec}')

        seq_mask = tf.cast(tf.equal(self.features[f"clk_doc_id"], tf.constant(0, tf.int64)), tf.float32)
        future_mask = 1-tf.expand_dims(tf.linalg.band_part(
            tf.ones_like(tf.tile(
                tf.expand_dims(seq_mask[0], axis=-1),
                [1, tf.shape(seq_mask)[1]],
            )), num_lower=-1, num_upper=0
        ), axis=0)

        with tf.variable_scope(f"SequenceOp_TransformerPosEmb"):
            interestEmb = InterestUnit(unit_fcs, 40)
            trm_inp = seq_vec * interestEmb
            tf.summary.histogram("InterestEmb",interestEmb)
            tf.summary.histogram("trm_inp", trm_inp)

        #### Transformer Layer ####
        with tf.variable_scope(f"SequenceOp_TransformerLayer"):
            trm_oup = tf_attention.SASRecTransformer(
                trm_inps=seq_vec * interestEmb,
                trm_attri=seq_attri,
                attention_heads=4,
                casual_mask=future_mask,
                mode=self.mode
            )
            print(f"- trm_oup:{trm_oup}")
        
        #### TargetAttention Layer ####
        with tf.variable_scope(f"SequenceOp_TargetAttentionLayer"):
            target_oup = tf_attention.TargetAttention(
                target_vec=target_vec,
                seq_vec=trm_oup,
                seq_mask=seq_mask,
                mode=self.mode
            )
            print(f"- target_oup:{target_oup}")
            return target_oup
