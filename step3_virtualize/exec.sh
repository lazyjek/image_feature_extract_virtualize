#!/bin/bash

image_cluster_file='../step2_extract_feature/outputs/cluster_res.txt'
image_virtualize_dir='output'
threads=16
if [ $# ]; then
    source $1
fi
python main.py -i $image_cluster_file -o $image_virtualize_dir -t $threads
