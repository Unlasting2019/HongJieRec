record_dir=/home/tiejianjie/news_rec/data/rank_record_data
data_dir=/home/tiejianjie/news_rec/data/feature_data


cd `dirname $0` && rm -rf build/  &&\
rm -rf $record_dir && mkdir -p $record_dir && \

echo "\n====================== start compiler ======================\n"&&
cmake \
    -S./ \
    -Bbuild/ \
    -DCPP_SRC_DIR=/home/tiejianjie/news_rec/src/rank/data_process \
    -DCPP_PKG_DIR=/home/tiejianjie/news_rec/src/cpp_pkg \
    -DCMAKE_CXX_COMPILER=/home/tiejianjie/cpp_pkg/gcc-12.2.0/bin/g++ \
    -DCMAKE_C_COMPILER=/home/tiejianjie/cpp_pkg/gcc-12.2.0/bin/gcc \
    -DCMAKE_CXX_FLAGS="-std=c++23 -O3 -w -W -lpthread -ldl -Wall -Wextra -Werror -pedantic" && \

make \
    -Cbuild/ \
    -j`nproc` && 

for day in $(seq 24 29)
do
    ./build/rank_tfrecord\
        --data_dir $data_dir \
        --record_dir $record_dir  \
        --record_size 512 \
        --thread_num 40 \
        --merge_day $day 
done
