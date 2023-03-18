cd `dirname $0` 

echo "\n====================== start nn_train ======================"&&
CUDA_VISIBLE_DEVICES=0 python -u -B -W ignore nn_train.py $1 $2   &&

echo "\n====================== start nn_infer ======================"&&
CUDA_VISIBLE_DEVICES=0 python -u -B -W ignore nn_infer.py $1 $2   &&

echo "\n====================== start eval ======================"&&
g++ eval.cpp -std=c++17 -O3 && \
    ./a.out \
        /home/tiejianjie/news_rec/data/feature_data/test_info.csv \
        tmp/$2.csv \
        $2 && rm -rf ./a.out
