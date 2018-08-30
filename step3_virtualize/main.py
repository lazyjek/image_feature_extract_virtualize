# -*- coding:utf-8 -*-
from virtualize import virtualize_image
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--inputs', default = '../step2_extract_feature/outputs/cluster_res.txt', required=False, type=str, help='image cluster_id file')
    parser.add_argument('-o', '--outputs', default = 'output', required=False, type=str, help='output directory')
    parser.add_argument('-t', '--threads', default = 16, required=False, type=int, help='number of threads')

    a = parser.parse_args()
    virtualize_image(inputs=a.inputs, outputs=a.outputs, num_threads=a.threads)
