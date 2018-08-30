#!/bin/bash
#rm data/image_list.*
source $1
file=video.pictures.$image_file_dir
gunzip $file.gz

mkdir $image_file_dir
split -l 20000 $file -d -a 3 $image_file_dir/image_list.

gzip $file
