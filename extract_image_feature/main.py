# -*- coding:utf-8 -*-
from feature import ImageFeature
from cluster import ImageKMeans
import argparse
import os,sys
import multiprocessing

def run_cluster(train=False, # whether to train or cluster
        inputs = 'features', # pkl directory / file / fnmatch
        model_path = 'checkpoint/kmeans.ckpt', # checkpoint file, whole path
        outputs = 'output/cluster_res.txt', # or [None], when doing clustering, you have to provide outputs as outputting path.
        num_clusters = 5,
        kmeans_plus_plus_num_retries = 2,
        num_steps = 100,
        observe_steps = 10,
        batch = 200):
    if train == False:
        """ cluster process
        Args:
            inputs
            outputs
            batch
            model_path @can be ignored"""
        k = ImageKMeans(inputs = inputs,
                model_path = model_path,
                outputs = outputs)
        k.cluster(batch)
        del k
    else:
        """ train process
        Args:
            inputs
            model_path
            num_clusters, kmeans_plus_plus_num_retries, num_steps,
            observe_steps @can be ignored"""
        k = ImageKMeans(inputs = inputs,
                model_path = model_path,
                outputs = None,
                num_clusters = num_clusters,
                kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries)
        k.save_kmeans_model(num_steps=num_steps,
                        observe_steps = observe_steps)
        del k

def run_feature(io_list = [('features/01','01.pkl'), ('features/02','02.pkl')],
        output_directory = 'features',
        batch  = 2):
    f = ImageFeature(image_feature_dir = output_directory, extract_image_batch = batch)
    io_iter = iter(io_list)
    force_exit = False
    while force_exit == False:
        try:
            inputs, outputs = next(io_iter)
        except StopIteration:
            force_exit = True
            break
        print ('input is:{}, output is:{}'.format(inputs, os.path.join(output_directory,outputs)))
        f.extract_image_feature(inputs, outputs)
    del f

if __name__ == '__main__':

    ''' initialize Parameters and Class Object '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # image_feature
    parser.add_argument('stage', type=str, help= \
            "[ClusterTrain]:\n\trequired:inputs, num_clusters, num_steps\n\tappendence: model_path, kpp, observe\n----------------------------------------------------\n[Cluster]:\n\trequired: inputs, outputs, batch\n\tappendence: model_path\n----------------------------------------------------\n[Feature]:\n\trequired: inputs, outputs, batch")
    parser.add_argument('-i', '--inputs', default = 'test_images', required=False, type=str, help='Directory images are in. Searches recursively.')
    parser.add_argument('-b', '--batch', default = 20, required=False, type=int, help='Batch Size of Predicted Image')
    parser.add_argument('-o', '--outputs', default = 'features', required=False, type=str, help='Directory feature pkls are in')
    parser.add_argument('-k', '--num_clusters', default = 100, required=False, type=int, help='cluster number for k-means model')
    parser.add_argument('-s', '--num_steps', default = 100, required=False, type=int, help='iteration number when training k-means model')
    parser.add_argument('-m', '--model_path', default = 'checkpoint/kmeans.ckpt', required=False,type=str,help='checkpoint file path')
    parser.add_argument('--kpp', default = 5, required=False, type=int, help='parameters for kmeans_plus_plus')
    parser.add_argument('--observe', default = 10, required=False, type=int, help='observe steps for training k-means model')
    a = parser.parse_args()

    if a.stage == 'ClusterTrain':
        """ train and save k-means model
        Args:
            stage [ClusterTrain]
            inputs: it is a directory for training data.
                |--inputs
                   |--1.pkl
                   |--2.pkl
                   |--3.pkl
                   |-- ...
            model_path: model save path
                for example, [checkpoint/kmeans.ckpt]
            num_clusters, num_steps
            kpp, observe @can be ignored """
        run_cluster(train=True,inputs = a.inputs,
        model_path = a.model_path,
        num_clusters = a.num_clusters,
        kmeans_plus_plus_num_retries = a.kpp,
        num_steps = a.num_steps,
        observe_steps = a.observe)
    elif a.stage == 'Cluster':
        """ cluster for pkl files
        Args:
            stage [Cluster]
            inputs: it is a directory for datas that will be clustered.
                |--inputs
                   |--1.pkl
                   |--2.pkl
                   |--3.pkl
                   |-- ...
            outputs: it is a path, not a directory
                for example: [output/cluster_res.txt]
            batch
            model_path @ can be ignored"""
        run_cluster(inputs = a.inputs,
                model_path = a.model_path,
                outputs = a.outputs,
                batch = a.batch)
    elif a.stage == 'Feature':
        """ extract image feature
        Args:
            stage [Feature]
            inputs:
                inputs should be a directory
                |--inputs
                   |--dir1
                      |--1.jpg
                      |--2.jpg
                      |-- ...
                   |--dir2
                      |--1.jpg
                      |--2.jpg
                      |-- ...
            outputs:
                outputs should also be a directory
                |--outputs
                   |--dir1.pkl
                   |--dir2.pkl
            batch """
        io_list = [(os.path.join(a.inputs.rstrip('/'),p), p + '.pkl') for p in os.listdir(a.inputs)]
        run_feature(io_list, a.outputs, a.batch)
    else:
        print ('stage ERROR!')
        exit()
