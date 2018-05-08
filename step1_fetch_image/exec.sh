#!/bin/bash

image_url_dir='url/data'
image_file_dir='images'
image_zip_dir='compress'
threads=16
process=("Spider Tar")

if [ $# -eq 2 ]; then
    source $1
    process=(`echo $2 | sed 's/,/ /g'`)
fi

for proc in ${process[@]}; do
    if [ $proc"x" == "Spiderx" ]; then
        python main.py $proc -i $image_url_dir -o $image_file_dir -t $threads
    fi
    if [ $proc"x" == "Tarx" ]; then
        python main.py $proc -i $image_file_dir -o $image_zip_dir -t $threads
    fi
done
