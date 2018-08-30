# -*- coding: utf-8 -*-
"""
File: cluster.py
Author: qixiucao
Email: jennifer.cao@wisc.edu
"""
import _pickle as pickle
import glob
import tensorflow as tf
import numpy as np
import fnmatch
import os
from tensorflow.contrib.factorization import KMeans
from tensorflow.python.framework import ops
from tensorflow.contrib.factorization.python.ops.clustering_ops \
        import CLUSTERS_VAR_NAME,_InitializeClustersOpFactory,COSINE_DISTANCE,nn_impl

class ImageKMeans(KMeans):
    def __init__(self,inputs = 'features',
            model_path = 'checkpoint/kmeans.ckpt',
            outputs = None, # output/cluster_res.txt
            feature_dims = 1536,
            num_clusters = 25,
            initial_clusters = 'kmeans_plus_plus',
            kmeans_plus_plus_num_retries = 5,
            random_seed = 0):

        # initialize kmeans model path.
        self.model_path = model_path
        try: os.makedirs('/'.join(model_path.split("/")[:-1]))
        except: pass

        # initialize outputs.
        self.outputs = outputs
        if outputs != None:
            try: os.makedirs('/'.join(outputs.split('/'[:-1])))
            except: pass

        # initialize input path
        def init_inputs(inputs=inputs):
            feat_dict = {}
            if inputs.endswith('.pkl'):
                print ("pklfile is: {}".format(inputs))
                try: feat_dict.update(pickle.load(open(inputs, 'rb')))
                except: print ('load_image_feature[%s] must provide ABSOLUTE PATH!'%inputs); exit()
            elif '*' in inputs:
                pklpaths = [ f for f in glob.iglob(inputs) if fnmatch.fnmatch(f, '*.pkl')]
                for p in pklpaths:
                    print ("pklfile is: {}".format(p))
                    try:feat_dict.update(pickle.load(open(p, 'rb')))
                    except: print ('load_image_feature[%s] EMPTY'%inputs); exit()
            else:
                pklpaths = [ os.path.join(d, filename) \
                        for d,_,fList in os.walk(inputs) \
                        for filename in fList if fnmatch.fnmatch(filename, '*.pkl')]
                for p in pklpaths:
                    print ("pklfile is: {}".format(p))
                    try:feat_dict.update(pickle.load(open(p, 'rb')))
                    except: print ('load_image_feature[%s] EMPTY'%inputs); exit()
            items = feat_dict.items()
            return [i[0] for i in items], np.asarray([i[1] for i in items])
        self.image_names, self.inputs = init_inputs(inputs)

        # initialize kmeans_x, fetch_kmeans, kmeans_session
        self.kmeans_x = tf.placeholder(tf.float32, shape=[None, feature_dims])
        self.kmeans_session, self.fetch_kmeans = None, []
        self.init_kmeans(num_clusters = num_clusters,
            initial_clusters = initial_clusters,
            random_seed = random_seed,
            kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries)

    def training_graph(self):
      """Generate a training graph for kmeans algorithm.
      """
      # Implementation of kmeans.
      if (isinstance(self._initial_clusters, str) or
          callable(self._initial_clusters)):
        initial_clusters = self._initial_clusters
        num_clusters = ops.convert_to_tensor(self._num_clusters)
      else:
        initial_clusters = ops.convert_to_tensor(self._initial_clusters)
        num_clusters = ops.convert_to_tensor(initial_clusters.shape[0])

      inputs = self._inputs
      (cluster_centers_var, cluster_centers_initialized, total_counts,
       cluster_centers_updated,
       update_in_steps) = self._create_variables(num_clusters)
      init_op = _InitializeClustersOpFactory(
          self._inputs, num_clusters, initial_clusters, self._distance_metric,
          self._random_seed, self._kmeans_plus_plus_num_retries,
          self._kmc2_chain_length, cluster_centers_var, cluster_centers_updated,
          cluster_centers_initialized).op()
      cluster_centers = cluster_centers_var

      if self._distance_metric == COSINE_DISTANCE:
        inputs = self._l2_normalize_data(inputs)
        if not self._clusters_l2_normalized():
          cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

      all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
      if self._use_mini_batch:
        sync_updates_op = self._mini_batch_sync_updates_op(
            update_in_steps, cluster_centers_var, cluster_centers_updated,
            total_counts)
        assert sync_updates_op is not None
        with ops.control_dependencies([sync_updates_op]):
          training_op = self._mini_batch_training_op(
              inputs, cluster_idx, cluster_centers_updated, total_counts)
      else:
        assert cluster_centers == cluster_centers_var
        training_op = self._full_batch_training_op(
            inputs, num_clusters, cluster_idx, cluster_centers_var)

      return (all_scores, cluster_idx, scores, cluster_centers_initialized,
              init_op, training_op)

    def init_kmeans(self, num_clusters = 25,
            initial_clusters = 'kmeans_plus_plus',
            distance_metric = 'cosine',
            use_mini_batch = True,
            mini_batch_steps_per_iteration = 1,
            random_seed = 0,
            kmeans_plus_plus_num_retries = 5,
            kmc2_chain_length = 200):

        # load saved cluster centers.
        if self.outputs != None:
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            sess = tf.Session()
            saver.restore(sess, self.model_path)
            cluster_centers = sess.run(tf.get_default_graph().get_tensor_by_name(CLUSTERS_VAR_NAME+':0'))
            initial_clusters = tf.convert_to_tensor(cluster_centers)
            sess.close()
            del sess
        KMeans.__init__(self,self.kmeans_x, num_clusters, initial_clusters, distance_metric,use_mini_batch,mini_batch_steps_per_iteration, random_seed,kmeans_plus_plus_num_retries,kmc2_chain_length)
        training_graph = self.training_graph()
        if len(training_graph) > 6:
            (all_scores, cluster_index, scores, cluster_centers_initialized,
                    cluster_centers_var, init_op, train_op) = training_graph
        else:
            (all_scores, cluster_index, scores, cluster_centers_initialized,
                    init_op, train_op) = training_graph

        cluster_index =  cluster_index[0]
        avg_distance = tf.reduce_mean(scores)
        init_vars = tf.global_variables_initializer()
        self.fetch_kmeans = [train_op, avg_distance, cluster_index, scores]
        if self.kmeans_session != None:
            self.kmeans_session.close()
            del self.kmeans_session
        self.kmeans_session = tf.Session()
        self.kmeans_session.run(init_vars, feed_dict={self.kmeans_x:self.inputs})
        self.kmeans_session.run(init_op, feed_dict={self.kmeans_x:self.inputs})

    def save_kmeans_model(self, num_steps=1000, observe_steps = 10):
        for i in range(1, num_steps + 1):
            _, d, idx, s = self.kmeans_session.run(self.fetch_kmeans, feed_dict={self.kmeans_x: self.inputs})
            if i == 1:
                print ("Start Avg Distance: %f, Scores: %f" % (d, s[0].sum()))
            if i % observe_steps == 0 or i == 1:
                print("Step %i, Avg Distance: %f, Scores: %f" % (i, d, s[0].sum()))
        print("Final Avg Distance: %f, Total Scores: %f" % (d, s[0].sum()))

        """ unit test
        try: os.makedirs('unit_test')
        except: pass
        with open('unit_test/kmeans_train.txt', 'w') as f:
            for i in idx:
                print (idx[i], file = f)
        """
        saver = tf.train.Saver()
        saver.save(self.kmeans_session, self.model_path)

    def cluster(self, batch=100):
        inputsL = [self.inputs[i:i+batch] for i in range(0,self.inputs.shape[0],batch)]
        image_namesL = [self.image_names[i:i+batch] for i in range(0,len(self.image_names),batch)]
        try: os.makedirs('/'.join(self.outputs.split('/')[:-1]))
        except: pass
        f = open(self.outputs, 'w')
        for ii in range(len(inputsL)):
            inputs = inputsL[ii]
            image_names = image_namesL[ii]
            _, d, idx, s = self.kmeans_session.run(self.fetch_kmeans, feed_dict = {self.kmeans_x:inputs})

            # paths [sample1_vid, sample2_vid, sample3_vid, ...]
            # idx [sample1_clusterid, sample2_clusterid, sample3_clusterid, ...]
            for i in range(len(image_names)):
                f.write('{}\t{}\n'.format(image_names[i], idx[i]))
        f.close()

    def __del__(self):
        if self.kmeans_session != None:
            self.kmeans_session.close()
            del self.kmeans_session

def unit_test_save_model():
    kmeans = ImageKMeans(inputs = 'features',
            model_path = 'checkpoint1/kmeans.ckpt',
            outputs = None,
            num_clusters = 5,
            kmeans_plus_plus_num_retries = 2)
    kmeans.save_kmeans_model(num_steps=10, observe_steps = 1)
    del kmeans

def unit_test_cluster():
    kmeans = ImageKMeans(inputs = 'features',
            model_path = 'checkpoint/kmeans.ckpt',
            outputs = 'output/cluster_res.txt')
    kmeans.cluster()
    del kmeans

if __name__ == '__main__':
    unit_test_cluster()
