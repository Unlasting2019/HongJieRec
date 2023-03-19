# Overview - 参考头条推荐架构, 针对用户行为序列进行特征计算
1. [SeqGenerate](https://github.com/Unlasting2019/HongJieRec/blob/master/src/feature/feature_main.py) - 使用spark生成用户点击和用户曝光序列（为了防止数据泄漏严格控制使用N-1的数据来构造N的序列）
2. [FeatureOps](https://github.com/Unlasting2019/HongJieRec/tree/master/src/rank/data_process/src/FeatureOps) - 使用c++构建特征算子MatchOp匹配算子/SequenceOp序列算子/TimeOp时间算子/Product笛卡尔积算子
3. [FeatureCross](https://github.com/Unlasting2019/HongJieRec/blob/master/src/rank/data_process/src/FeatureProfile.h#L104) - 对于每一条样本根据user_id取出对应的点击序列和曝光序列, 然后根据序列计算用户特征和用户Feed的交叉特征
4. [TFRecordDump](https://github.com/Unlasting2019/HongJieRec/blob/master/src/cpp_pkg/cpp_tfrecord/dump_tfrecord.h) - 将计算的样本以TFRecord格式保存, 不定长的特征若为长度为0则填充默认值
5. [ModelTrain](https://github.com/Unlasting2019/HongJieRec/blob/master/src/rank/tf_rank/nn_train.py) - 使用tf.data和tf.estimator训练模型

# Requirements
* CUDA 11.x, [Nvidia/Tensorflow 1.15](https://github.com/NVIDIA/tensorflow), python 3.8
* GNU-GCC 12.2.0, [cpp_fmt](https://github.com/fmtlib/fmt), protobuf-cpp-3.8

# Todo
- [ ] 学习完多线程相关知识实现，实现一个线程池。（目前开启多线程的方式不优雅）
- [ ] 学习完锁相关的知识后，读取数据开启多线程加速。（虽然目前读一张csv只需要2s，但考虑到未来大规模数据）
- [ ] 深入学习ranges这个库后，dump不定长特征时，删掉手动指定类型。（目前dump的时候要指定类型type，但类型type在编译时都可以被推导出来，没有必要。 ）

# CodeStructure
```
src/
├── cpp_pkg
│   ├── CharConv
│   ├── cpp_frame # 使用静态反射解析csv文件
│   ├── cpp_reflect 
│   ├── cpp_str
│   ├── cpp_tfrecord # 样本保存为tfrecord
│   ├── fast_float
│   ├── fmt 
│   └── protobuf-cpp-3.8.0.tar.gz
├── feature
│   ├── feature_main.py 
│   └── run.sh
├── rank
│   ├── data_process
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   ├── run.sh
│   │   └── src
│   │       ├── FeatureOps # 特征算子代码库
│   │       ├── FeatureProfile.h # 核心调用函数FeatureOps
│   │       └── FeatureStruct.h # 定义DataSource结构体
│   └── tf_rank 
│       ├── EstimatorDataOp.py
│       ├── ModelZoo # 模型网络图
│       ├── eval.cpp
│       ├── feature_columns.py
│       ├── nn_infer.py
│       ├── nn_train.py # trainer
│       ├── run.sh
│       └── tmp
│           └── tf_conf.py
└── utils.py
