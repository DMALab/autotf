from __future__ import division, print_function, absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

from base_model import BaseModel
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import os

class RandomForest(BaseModel):

    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "batch_size" : 1024,
        "num_epochs" : 25,
        "num_trees" : 10,
        "max_nodes":1000
    }

    def __init__(self, feature_num,classnum):
        self.feature_num = feature_num
        self.class_num = classnum

        self.model = None
        self.sess = tf.Session()

    def build_model(self):
        with tf.variable_scope("RandomForest"):

            # Input and Target data
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.feature_num])
            # For random forest, labels must be integers (the class id)
            self.labels = tf.placeholder(tf.int32, shape=[None])

    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        num_trees = param['num_trees']
        max_nodes = param['max_nodes']

        # Random Forest Parameters
        self.hparams = tensor_forest.ForestHParams(num_classes=self.class_num,
                                                   num_features=self.feature_num,
                                                   num_trees=num_trees,
                                                   max_nodes=max_nodes).fill()

        # Build the Random Forest
        self.forest_graph = tensor_forest.RandomForestGraphs(self.hparams)
        # Get training graph and loss
        self.train_op = self.forest_graph.training_graph(self.inputs, self.labels)
        self.loss = self.forest_graph.training_loss(self.inputs, self.labels)

        # Measure the accuracy
        self.infer_op, _, _ = self.forest_graph.inference_graph(self.inputs)
        self.correct_prediction = tf.equal(tf.argmax(self.infer_op, 1), tf.cast(self.labels, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        #metrics = [self.get_metric(metric) for metric in param["metrics"]]
        #self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]
        self.init_vars = tf.group(tf.global_variables_initializer(),
                                  resources.initialize_resources(resources.shared_resources()))
        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        X = feed_data["inputs"]
        Y = feed_data["labels"]
        totalbatch = int(len(X)/self.batch_size)

        for i in range(0,totalbatch):
            startindex = i*self.batch_size
            endindex = (i+1)*self.batch_size
            batch_xs = X[startindex:endindex]
            batch_ys = Y[startindex:endindex]
            yield { "batch_xs" : batch_xs, "batch_ys" : batch_ys }

    def train(self, feed_data):
        self.sess.run(self.init_vars)
        trainstep = 0
        totallen = len(feed_data["inputs"])

        for epoch in range(self.num_epochs):
            avg_cost = 0.0
            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                }
                _,loss, acc = self.sess.run([self.train_op,self.loss,self.accuracy], feed_dict=feed_dict)
                totalaccuracy += acc*len(batch["batch_xs"])
                avg_cost +=  loss
                trainstep = trainstep + 1
            avg_cost /= totallen
            totalaccuracy /= len(feed_data['inputs'])
            print("train_step"+"\t"+str(trainstep)+"\t"+"epoch:"+"\t"+str(epoch+1)+"\t"+"accuracy:"+"\t"+str(totalaccuracy)+"\t"+"loss:"+"\t"+str(avg_cost))

    def evaluate(self, feed_data):
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"],
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            totalaccuracy += acc * len(batch["batch_xs"])
            avg_loss += loss
        avg_loss /= totallen
        totalaccuracy /= len(feed_data['inputs'])

        res = {"accuracy":totalaccuracy,"loss":avg_loss}
        return res

    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return




