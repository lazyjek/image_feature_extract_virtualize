# -*- coding:utf-8 -*-
from get_image_file import save_images,tar_files
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('stage', type=str, help='[Spider] / [Tar]')
    parser.add_argument('-i', '--inputs', default = 'url/data', required=False, type=str, help='stage [Spider]: input is image_url directory;\nstage [Tar]: input is directory of images.\nSearches recursively.')
    parser.add_argument('-o', '--outputs', default = 'output', required=False, type=str, help='stage [Spider]: output is image directory;\nstage [Tar]: output is zip directory')
    parser.add_argument('-t', '--threads', default = 16, required=False, type=int, help='number of threads')

    a = parser.parse_args()
    f = None
    if a.stage == 'Spider':
        print ("input: {}, output:{}, Spider ...".format(a.inputs, a.outputs))
        f = save_images
    elif a.stage == 'Tar':
        print ("input: {}, output:{}, compress ...".format(a.inputs, a.outputs))
        f = tar_files
    else:
        print ('stage ERROR!')
        import sys
        sys.exit()
    f(outputs=a.outputs, inputs=a.inputs, num_threads=a.threads)

