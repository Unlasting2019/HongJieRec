sciprt_list="
    WideTrmMTL7
    WideTrmMTL8
"

for model_name in $sciprt_list;
do
    for i in $(seq 0 4);
    do
        echo "$i $model_name start_train"
        sh scripts/train.sh $1 $model_name
    done

    cat log | grep "${model_name}_auc\|${model_name}_ugauc\|${model_name}_dgauc"  > logs/${model_name}

done

echo "experiment done"
