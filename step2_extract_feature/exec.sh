#!/bin/bash
num_clusters=200
kmeans_plus_plus_num_retries=5
observe_steps=20
num_steps=200
original_image_dir='../step1_fetch_image/images'
feature_dump_dir='../step2_extract_feature/06-25_07-02_features'
checkpoint_file='../step2_extract_feature/checkpoint_200/kmeans_model.ckpt'
extract_image_feature_batch=25
cluster_batch=200
cluster_res='output/06-25_07-02.res'

if [ ! -s 'inception_resnet_v2_2016_08_30.ckpt' ]; then
    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz && tar -zxvf inception_resnet_v2_2016_08_30.tar.gz
    rm -f inception_resnet_v2_2016_08_30.tar.gz
fi

process=("Cluster")
if [ $# -eq 2 ];then
    source $1
    process=(`echo $2 | sed 's/,/ /g'`)
fi
if [ $# -eq 1 ];then
    feature_dump_dir=$1
fi

#echo "start to wait .."
#sleep 180m
for proc in ${process[@]}; do
    if [ $proc"x" == "Featurex" ]; then
        python main.py $proc -i $original_image_dir -o $feature_dump_dir -b $extract_image_feature_batch
    fi
    if [ $proc"x" == "ClusterTrainx" ]; then
        python main.py ClusterTrain -i $feature_dump_dir -m $checkpoint_file -k $num_clusters \
            -s $num_steps --kpp=$kmeans_plus_plus_num_retries --observe=$observe_steps
    fi
    if [ $proc"x" == "Clusterx" ]; then
        # when "*" not in feature_dump_dir
        if [[ $feature_dump_dir =~ "*" ]]; then
            # match *, result is XXXXX*XXXX
            echo "Cluster for a Bunch of files: $feature_dump_dir"
            python main.py $proc -i "$feature_dump_dir" -o $cluster_res -b $cluster_batch -m $checkpoint_file
        else
            # not match *, result is ""
            cluster_res='output/'`echo $feature_dump_dir | awk -F '/' '{print $NF}'`'.res'
            echo "Cluster for Directory: $feature_dump_dir"
            cluster_dirs=(`ls $feature_dump_dir`)
            for i in ${cluster_dirs[@]}; do
                python main.py Cluster -i "$feature_dump_dir/$i" -o "$cluster_res.$(echo $i | cut -d . -f 1)" -b $cluster_batch -m $checkpoint_file
            done
            cat $cluster_res.* | awk -F ' ' '{split($1,a,".");n=split(a[1],b,"\\/");printf("%s|%s\n",b[n],$2)}' > $cluster_res && rm $cluster_res.*
            #cat $cluster_res.* > $cluster_res && rm $cluster_res.*
        fi
    fi
done
