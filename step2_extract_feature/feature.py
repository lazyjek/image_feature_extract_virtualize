# -*- coding: utf-8 -*-
"""
File: feature.py
Author: qixiucao
Email: jennifer.cao@wisc.edu
"""
import scipy.misc as misc
import _pickle as pickle
import tensorflow as tf
import numpy as np
import fnmatch
import os
import glob
import time
from inception_resnet_v2 import inception_resnet_v2_arg_scope,inception_resnet_v2
import PIL.Image as Image

class ImageFeature(object):
    def __init__(self, image_feature_dir = 'features',
            extract_image_batch = 20,
            **kw):
        """ Initiate ImageFeature Class
          Args:
              image_feature_dir: image feature pickle file directory,
                    output of inception resnet v2 model.
              extract_image_batch: batch size when extracting image feature.
              ...
        """
        self.image_feature_dir = image_feature_dir

        # extract image feature parameters.
        # initialize placeholder, features, session
        def _init_inception_resnet_v2_session():
            height, width, channels, checkpoint_file = 299, 299, 3, 'inception_resnet_v2_2016_08_30.ckpt'
            slim = tf.contrib.slim
            x = tf.placeholder(tf.float32, shape=(None, height, width, channels))
            arg_scope = inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
               logits, end_points = inception_resnet_v2(x, is_training=False, num_classes=1001)
               features = end_points['PreLogitsFlatten']
            sess  = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)
            return x, features, sess, height, width
        self.extract_image_x, self.image_features, self.extract_image_session, \
                self.height, self.width = _init_inception_resnet_v2_session()

        # initialize image feature.
        self.extract_image_batch = extract_image_batch

        for k,w in kw.items():
            setattr(self, k, w)

    def _init_images(self, inputs, convert=False):
       """Initialize Input Image Paths and Image Array.
            This is a inner function that should not be used outside of this Class.
            Every time you do this, images will be recalculated based on the input image directory.
        Return:
          images: image dict: {path, image array}
       """
       def _preprocess_batch(img):
           return ([i[0] for i in img], np.asarray([i[1] for i in img]))

       if convert == True:
            images = [[
                    os.path.join(d, filename),
                    misc.fromimage(Image.open(os.path.join(d,filename)).convert('RGB').resize((self.height, self.width), Image.ANTIALIAS))
                        ] for d,_,fList in os.walk(inputs) \
                        for filename in fList \
                        if fnmatch.fnmatch(filename, '*.jpg')]
       else:
            images = [[
                    os.path.join(d, filename),
                    misc.fromimage(Image.open(os.path.join(d,filename)))
                        ] for d,_,fList in os.walk(inputs) \
                        for filename in fList \
                        if fnmatch.fnmatch(filename, '*.jpg')]
       return [_preprocess_batch(images[i:i+self.extract_image_batch]) for i in range(0,len(images),self.extract_image_batch)]

    def extract_image_feature(self, inputs = 'test_images',
            outputs = 'test.pkl', convert = False, debug = False, monitor = False):
        """Extract Image Feature by Inception Resnet V2 Model.
          This function is the Interface of this Class.
          Every time you use this function, it will dump image feature from
          the input directory to the output pickle files.

         Args:
           inputs: a directory that includes jpg image files.
             This directory can be a parent folder, so long as there exists any *.jpg
           files. For example, it can be [dir] or [dir/XXX] which includes a
           [dir/XXX/*.jpg] file.

           outputs: name of output pickle file.
              Output directory will be [output_pkl_dir/outputs]

           convert: bool. Whether to convert input images.
             If it is true, then the input pictures will be convert into 'RGB' mode with
           a size of (299, 299).

           debug: bool. Whether you want to stop the process and observe
              key, values from feature dict.
             It should never be true if you do not want to debug your code.

           monitor: bool. Whether to provide time consuming details for each step.
             If it is true, then you can observe how long does it take to:
                  1. initialize image matrix from local paths.
                  2. predicting image features from image matrix.
                  3. dump pickle files from feature dict.
        """
        feat_dict = {}
        if monitor == True:
           start = time.time()
        images = self._init_images(inputs, convert)
        if monitor == True:
            end = time.time()
            print ('TIME for initialize image matrix from local paths: %.2f s'%(end - start))
            start = end

        for paths, img in images:
           feat = self.extract_image_session.run(self.image_features, feed_dict={self.extract_image_x:img})
           feat_dict.update({paths[i]:feat[i] for i in range(len(paths))})

#        feat_dict = {
#            paths[i]:self.extract_image_session.run(self.image_features, feed_dict={self.extract_image_x:img})[i] \
#                    for paths, img in images \
#                    for i in range(len(paths))
#            }

        if monitor == True:
            end = time.time()
            print ('TIME for predicting image features from image matrix: %.2f s'%(end - start))
            start = end

        if debug == True:
            for k in feat_dict:
                print ('path: {}, feature shape: {}, feature: {}'.format(k, feat_dict[k].shape, feat_dict[k]))
                exit()

        # dump feature dict.
        try: os.makedirs(self.image_feature_dir)
        except: pass
        exp_pkl = open(os.path.join(self.image_feature_dir, outputs), 'wb')
        data = pickle.dumps(feat_dict)
        exp_pkl.write(data)
        exp_pkl.close()
        print ("finish output is {}".format(os.path.join(self.image_feature_dir, outputs)))
        if monitor == True:
            end = time.time()
            print ('dump pickle files from feature dict: %.2f s'%(end - start))

    def load_image_feature(self, inputs=None):
        """
        Args:
            pklfile: should be the directory or pklfile's absolute path.
                it is independent with self.image_feature_dir !
        """
        if inputs == None:
            inputs = self.image_feature_dir
        feat_dict = {}
        if inputs.endswith('.pkl'):
            try: feat_dict.update(pickle.load(open(inputs, 'rb')))
            except: print ('load_image_feature[%s] must provide ABSOLUTE PATH!'%inputs); exit()
        elif '*' in inputs:
            pklpaths = [ f for f in glob.iglob(inputs) if fnmatch.fnmatch(f, '*.pkl')]
            for p in pklpaths:
                try:feat_dict.update(pickle.load(open(p, 'rb')))
                except: print ('load_image_feature[%s] EMPTY'%inputs); exit()
        else:
            pklpaths = [ os.path.join(d, filename) \
                    for d,_,fList in os.walk(inputs) \
                    for filename in fList if fnmatch.fnmatch(filename, '*.pkl')]
            for p in pklpaths:
                try:feat_dict.update(pickle.load(open(p, 'rb')))
                except: print ('load_image_feature[%s] EMPTY'%inputs); exit()
        items = feat_dict.items()
        return [i[0] for i in items], np.asarray([i[1] for i in items])

    def __del__(self):
        if self.extract_image_session != None:
            self.extract_image_session.close()
            del self.extract_image_session

def unit_test_extract_feature():

    """ test
        1. extract features from *.jpg files and save as *.pkl files.
        2. load dict from *.pkl files.
    """
    imageFeature = ImageFeature(image_feature_dir='output/test', extract_image_batch=2)
    imageFeature.extract_image_feature(inputs = 'test_images/02', outputs = '02.pkl', monitor = True)
    imageFeature.extract_image_feature(inputs = 'test_images/01', outputs = '01.pkl', monitor = True)
    imageFeature.extract_image_feature(inputs = 'test_images', outputs = 'all.pkl')
    paths, datas = imageFeature.load_image_feature()
    for i in range(len(paths)):
        print (paths[i], datas[i])
    del imageFeature

if __name__ == '__main__':
    unit_test_extract_feature()
