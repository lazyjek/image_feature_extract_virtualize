#!/bin/bash
rm data/image_list.*
split -l 5000 "video.pictures" -d -a 3 data/image_list.
