# 抽取图像特征pipeline

**共分为三步**
1. (step1) 图片抓取（通过url抓取图片）
2. (step2)特征抽取（inception_resnet_v2）
   （1）paper：http://arxiv.org/abs/1602.07261
   （2）tensorflow官方模型文件：http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
3. (step2)图片聚类

**软件版本依赖**
1. python 3.6.4
2. tensorflow 1.7.0

**代码使用说明**
1. 将输入文件（文件内容:图片名\tab图片url，文件格式：gz，命名为：video.pictures.输入文件后缀.gz）放入step1_fetch_image/url目录。
2. 所有配置均在根目录的run.conf中。将代码克隆到目录后，修改run.conf如下:
         `ROOT_DIR=代码库的根目录`
         `image_file_dir=输入文件后缀` 注意输入文件后缀与步骤1中video.pictures.输入文件后缀.gz中的内容保持一致。
3. 执行run.sh即可，输出文件在step2_extract_feature/output目录中，输出格式为【输入文件后缀_features.res】
