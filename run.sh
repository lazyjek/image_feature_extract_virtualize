#!/bin/bash
root_dir=$(pwd)
steps=("Spider,Feature,ClusterTrain,Cluster")
config=$root_dir/run.conf
dirL=(`ls | grep -E '^step[0-9]+_'`)

for _dir in ${dirL[@]}; do
   cd $_dir
   bash exec.sh $config $steps
   cd $root_dir
done
