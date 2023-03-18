import tensorflow as tf
import feature_columns as fc
import time
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tmp import tf_conf
from tqdm import tqdm


class EstimatorDataOp:
    def __init__(self, record_size):
        self.feature_columns = {c['field']:fc.get_feature_column(**c) for c in tf_conf.x_col + tf_conf.y_col}
        self.record_size = record_size
        self.parse_x_col = [c['field'] for c in tf_conf.x_col]
        self.parse_y_col = [c['field'] for c in tf_conf.y_col]

    def parse_single_record(self, tf_exp, columns):
        ctx_spec = {}; seq_spec = {}
        for col in columns:
            self.feature_columns[col].parse_record_op(
                ctx_spec, seq_spec, self.record_size)
        
        ctx_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            tf_exp,
            context_features=ctx_spec,
            sequence_features=seq_spec
        )

        return {**ctx_parsed, **seq_parsed}

    def batch_fn(self, f_parsed, columns):
        for col in columns:
            self.feature_columns[col].batch_fn(f_parsed)
        return f_parsed

    def record_input_fn(self, record_list, batch_size, cpu_num=40, gpu_num=2):
        auto_tune = tf.data.experimental.AUTOTUNE
        return tf.data.Dataset.from_tensor_slices(
            record_list
        ).apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=cpu_num,
                sloppy=True
            )
        ).apply(
            tf.contrib.data.map_and_batch(
                lambda tf_exp : self.parse_single_record(
                    tf_exp,
                    self.parse_x_col + self.parse_y_col,
                ), batch_size=batch_size, num_parallel_calls=cpu_num
            )
        ).map(
            lambda parsed_f : self.batch_fn(
                parsed_f,
                self.parse_x_col + self.parse_y_col
            )
        ).prefetch(gpu_num)
   
    def infer_input_fn(self):
        tf_exp = tf.placeholder(
            dtype=tf.string,
            shape=[None],
            name='tf_exp'
        )
        #### parse feature ####
        ctx_spec = {}; seq_spec = {}
        for col in self.parse_x_col:
            self.feature_columns[col].parse_record_op(
                ctx_spec, seq_spec, self.record_size)
        ctx_parsed, seq_parsed, _ = tf.io.parse_sequence_example(
            tf_exp,
            context_features=ctx_spec,
            sequence_features=seq_spec
        )
        f_parsed = self.batch_fn({**ctx_parsed, **seq_parsed}, self.parse_x_col)

        return tf.estimator.export.ServingInputReceiver(f_parsed, {'tf_exp':tf_exp})
