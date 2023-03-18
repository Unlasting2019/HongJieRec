cd $(cd "$(dirname "$0")"; pwd) && 
spark-submit \
    --driver-memory 20G \
    --executor-memory 1G \
    --total-executor-cores 40 \
    --executor-cores 1 \
    feature_main.py $1

rm -rf spark-warehouse
