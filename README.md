# Overview
* 参考头条推荐架构, 针对用户点击序列和用户曝光序列进行特征计算。
* 使用pandas或者Spark生成点击和曝光序列以csv保存到磁盘上. csv有两个field: user_id & SeqStruct，SeqStruct是一个四元组Tuple - doc_id/watch_time/ctx_exposedTime/ctx_refreshTimes
* 对于每一条样本，根据user_id取出用户点击序列和用户曝光序列, 然后使用c++构建特征算子MatchOp匹配算子/SequenceOp序列算子/TimeOp时间算子/Product笛卡尔积算子, 来计算用户特征和用户Feed的交叉特征
* 最后将算好的特征以TFRecord格式序列化到磁盘，不定长特征在存的时候不pad到定长，解析的时候用tf.io.VarLenFeature。训练时根据是否需要可以转Dense或者直接reduce，DIN/DIEN/Transformer等模型就需要转Dense，不需要转dense可以使用tf.gather & tf.math.segment_sum完成reduce（比如当前doc和历史点击doc的笛卡尔积序列特征）
* 后续使用tf.data.TFRecordDataset读取数据，tf.estimator训练模型。

# Requirements
* CUDA 11.x, [Nvidia/Tensorflow 1.15](https://github.com/NVIDIA/tensorflow), python 3.8
* GNU-GCC 12.2.0, [cpp_fmt](https://github.com/fmtlib/fmt), protobuf-cpp-3.8
