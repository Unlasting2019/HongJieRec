import tensorflow as tf



class SingleColumn:
    def __init__(self, field):
        self.field = field

    def parse_record_op(self, ctx_spec, seq_spec, record_size):
        with tf.name_scope(f"parse_batch_record/SingleColumn_{self.field}"):
            ctx_spec[self.field] = tf.io.FixedLenFeature(shape=[record_size], dtype=tf.int64)

    def batch_fn(self, f_parsed):
        f_parsed[self.field] = tf.reshape(f_parsed[self.field], [-1])


class KeyValueColumn:
    def __init__(self, field):
        self.key_field, self.val_field = f"{field}_key", f"{field}_val"

    def parse_record_op(self, ctx_spec, seq_spec, record_size):
        with tf.name_scope(f"parse_batch_record/KeyValueColumn_{self.key_field}_{self.val_field}"):
            seq_spec[self.key_field] = tf.io.VarLenFeature(dtype=tf.int64)
            seq_spec[self.val_field] = tf.io.VarLenFeature(dtype=tf.float32)

    def batch_fn(self, f_parsed):
        f_parsed[self.key_field] = tf.sparse.reshape(f_parsed[self.key_field], [-1, tf.shape(f_parsed[self.key_field])[2]])
        f_parsed[self.val_field] = tf.sparse.reshape(f_parsed[self.val_field], [-1, tf.shape(f_parsed[self.val_field])[2]])



class SequenceColumn:
    def __init__(self, field, dtype, sparse=False):
        self.field = field
        self.dtype = tf.float32 if dtype == 'float' else tf.int64
        self.sparse = sparse

    def parse_record_op(self, ctx_spec, seq_spec, record_size):
        with tf.name_scope(f"parse_batch_record/SequenceColumn_{self.field}"):
            seq_spec[self.field] = tf.io.VarLenFeature(dtype=self.dtype)

    def batch_fn(self, f_parsed):
        if not self.sparse:
            f_parsed[self.field] = tf.sparse.to_dense(f_parsed[self.field])
            f_parsed[self.field] = tf.reshape(f_parsed[self.field], [-1, tf.shape(f_parsed[self.field])[2]])
        else:
            f_parsed[self.field] = tf.sparse.reshape(f_parsed[self.field], [-1, tf.shape(f_parsed[self.field])[-1]])


class NumericColumn:
    def __init__(self, field):
        self.field = field

    def parse_record_op(self, ctx_spec, seq_spec, record_size):
        with tf.name_scope(f"parse_batch_record/NumericColumn_{self.field}"):
            ctx_spec[self.field] = tf.io.FixedLenFeature(shape=[record_size], dtype=tf.float32)

    def batch_fn(self, f_parsed):
        f_parsed[self.field] = tf.reshape(f_parsed[self.field], [-1, 1])



def get_feature_column(
    fieldColumn,
    field,
    sparse=False,
    dtype=None,
):
    if fieldColumn == 'Single':
        return SingleColumn(field)
    elif fieldColumn == 'Numeric':
        return NumericColumn(field)
    elif fieldColumn == 'KeyValue':
        return KeyValueColumn(field)
    elif fieldColumn == 'Sequence':
        return SequenceColumn(field, dtype, sparse)
    else:
        raise ValueError(f'feature_column:{fieldColumn} not found')
