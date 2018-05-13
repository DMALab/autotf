#-*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from helper import *
import tensorflow as tf
import pickle
import numpy as np
import time

class Vgg16(BaseModel):
    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25,
        "keep_prob":0.75
    }

    def __init__(self,classnum):
        self.class_num = classnum
        self.model = None
        self.sess = tf.Session()
        self.scope = {}
        self.summary = []
    def conv2d(self,layer_name,inputs, out_channels, kernel_size, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs

    def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
        with tf.name_scope(layer_name):
            return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                      alpha=alpha, beta=beta)

    def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)

    def dropout(self, layer_name, inputs, keep_prob):
        # dropout_rate = 1 - keep_prob
        with tf.name_scope(layer_name):
            return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)

    def bn(self, layer_name, inputs, epsilon=1e-3):
        with tf.name_scope(layer_name):
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            inputs = tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=None,
                                               scale=None, variance_epsilon=epsilon)
            return inputs

    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
            return inputs

    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)

        self.conv1_1 = self.conv2d("conv1_1",self.inputs,64,3)
        self.conv1_2 = self.conv2d("conv1_2",self.conv1_1, 64,3)
        self.pool1 = self.max_pool('pool1',self.conv1_2,pool_size=2,strides=2)
        #112*112*64

        self.conv2_1 = self.conv2d("conv2_1",self.pool1, 128,3)
        self.conv2_2 = self.conv2d( "conv2_2",self.conv2_1, 128,3)
        self.pool2 = self.max_pool("pool2",self.conv2_2,pool_size=2,strides=2)
        #56*56*128

        self.conv3_1 = self.conv2d("conv3_1",self.pool2, 256,3)
        self.conv3_2 = self.conv2d("conv3_2",self.conv3_1, 256,3)
        self.conv3_3 = self.conv2d("conv3_3",self.conv3_2, 256, 3)
        self.pool3 = self.max_pool("pool3",self.conv3_3,pool_size=2,strides=2)
        #28*28*256

        self.conv4_1 = self.conv2d("conv4_1",self.pool3, 512, 3)
        self.conv4_2 = self.conv2d("conv4_2",self.conv4_1, 512, 3)
        self.conv4_3 = self.conv2d("conv4_3",self.conv4_2, 512, 3)
        self.pool4 = self.max_pool("pool4",self.conv4_3, pool_size=2,strides=2)
        #14*14*512

        self.conv5_1 = self.conv2d("conv5_1",self.pool4, 512, 3)
        self.conv5_2 = self.conv2d("conv5_2",self.conv5_1, 512, 3)
        self.conv5_3 = self.conv2d("conv5_3",self.conv5_2, 512, 3)
        self.pool5 = self.max_pool( 'pool5',self.conv5_3,pool_size=2,strides=2)
        #7*7*512

        self.fc6 = self.fc("fc6",self.pool5,4096)  # 25088 = 7*7*512

        self.relu6 = tf.nn.dropout(self.fc6, self.keep_prob)

        self.fc7 = self.fc("fc7",self.relu6,4096)

        self.relu7 = tf.nn.dropout(self.fc7, self.keep_prob)

        self.pred = self.fc("fc8",self.relu7, self.class_num)


    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        # 定义交叉熵损失函数
        self.keep_prob_value = param["keep_prob"]

        loss_fun = param["loss"]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))


        optimizer = param["optimizer"]
        self.learning_rate = param["learning_rate"]

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        X = feed_data["inputs"]
        Y = feed_data["labels"]
        totalbatch = int(len(X)/self.batch_size)+1
        if (totalbatch * self.batch_size == len(X)):
            totalbatch = totalbatch - 1

        for i in range(0,totalbatch):
            startindex = i*self.batch_size
            endindex = (i+1)*self.batch_size
            batch_xs = X[startindex:endindex]
            batch_ys = Y[startindex:endindex]
            yield { "batch_xs" : batch_xs, "batch_ys" : batch_ys }

    def train(self, feed_data):
        self.sess.run(tf.global_variables_initializer())
        trainstep = 0
        for epoch in range(self.num_epochs):
            avg_cost = 0.0
            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                    self.keep_prob: self.keep_prob_value,
                }
                _, loss, acc = self.sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
                totalaccuracy += acc*len(batch["batch_xs"])
                avg_cost +=  loss
                trainstep = trainstep + 1

            totalaccuracy /= len(feed_data['inputs'])
            print("train_step"+"\t"+str(trainstep)+"\t"+"epoch:"+"\t"+str(epoch+1)+"\t"+"accuracy:"+"\t"+str(totalaccuracy)+"\t"+"loss:"+"\t"+str(avg_cost))

    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return

    def evaluate(self, feed_data):
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"],
                self.keep_prob:self.keep_prob_value
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            totalaccuracy += acc * len(batch["batch_xs"])
            avg_loss += loss
        avg_loss /= totallen
        totalaccuracy /= len(feed_data['inputs'])

        res = {"accuracy":totalaccuracy,"loss":avg_loss}
        return res

    def predict(self, feed_data):
        res = []
        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"]
            }
            pred = self.sess.run(self.pred, feed_dict=feed_dict)
            res.extend(pred.tolist())

        return res

