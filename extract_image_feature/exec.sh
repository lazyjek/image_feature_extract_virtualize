#!/bin/bash
num_clusters=500
kmeans_plus_plus_num_retries=5
observe_steps=20
num_steps=500
# feature_dir must exist
original_image_dir='/home/jennifer/ImageFeature/fetch_image/04_27'
feature_dump_dir='features'
checkpoint_file='checkpoint/kmeans.ckpt'
extract_image_feature_batch=30
cluster_batch=200
cluster_res='output/cluster.res'
#process=("Feature ClusterTrain Cluster")
process=("Cluster")
process=("Feature")

if [ ! -e inception_resnet_v2_2016_08_30.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    tar -zxvf inception_resnet_v2_2016_08_30.tar.gz
    if [ $? -ne 0 ]; then
        echo "wget checkpoint ERROR!"
        exit
    fi
fi

if [ ! -e inception_resnet_v2.py ]; then
    wget https://github.com/cameronfabbri/Compute-Features/blob/master/nets/inception_resnet_v2.py
    if [ $? -ne 0 ]; then
        echo "wget model parser ERROR!"
        exit
    fi
fi

for proc in ${process[@]}; do
    if [ $proc"x" == "Featurex" ]; then
        python main.py $proc -i $original_image_dir -o $feature_dump_dir -b $extract_image_feature_batch
    fi
    if [ $proc"x" == "ClusterTrainx" ]; then
        python main.py ClusterTrain -i $feature_dump_dir -m $checkpoint_file -k $num_clusters \
            -s $num_steps --kpp=$kmeans_plus_plus_num_retries --observe=$observe_steps
    fi
    if [ $proc"x" == "Clusterx" ]; then
        cluster_dirs=(`ls $feature_dump_dir`)
        for i in ${cluster_dirs[@]}; do
            python main.py Cluster -i "$feature_dump_dir/$i" -o "$cluster_res.$(echo $i | cut -d . -f 1)" -b $cluster_batch -m $checkpoint_file
        done
    fi
done
