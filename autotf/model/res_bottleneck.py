# -*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from util import *

import tensorflow as tf
import pickle


class res_bottleneck(BaseModel):
    default_param = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-2,
        "batch_size": 100,
        "num_epochs": 25,
        "block_num": 5,
        "class_num": 10,
    }

    def __init__(self):

        self.sess = tf.Session()
        self.summary = []
        self.scope = {}

    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        #32*32*3

        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])

        self.net= self.conv2d('conv2d', self.inputs, 64, 3, padding="SAME")

        self.net = self.resnet_block("bottleneck1",self.net, self.block_num, bottleneck_size=16,out_channels=64)

        self.net = self.resnet_block("bottleneck2",self.net, 1, 32, 128,downsample=True)

        self.net = self.resnet_block("bottleneck3",self.net, self.block_num-1, 32, 128)

        self.net = self.resnet_block("bottleneck4",self.net, 1, 64, 256, downsample=True)

        self.net = self.resnet_block("bottleneck5",self.net, self.block_num - 1, 64, 256)

        # class number
        self.net = self.bn("bn4",self.net)
        self.net = tf.nn.relu(self.net)
        self.pred = self.fc("fully_connect",self.net,self.class_num)


    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME',batch_norm=True):
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
            if batch_norm:
                inputs = self.bn("batch_norm",inputs)
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs


    def bottleneck_block(self, inputs, nb_blocks, bottleneck_size, out_channels,
                            downsample=False, downsample_strides=2,
                            activation='relu', batch_norm=True, bias=True,
                            weights_init='variance_scaling', bias_init='zeros',
                            regularizer='L2', weight_decay=0.0001,
                            trainable=True, restore=True, reuse=False, scope=None,
                            name="ResidualBottleneck"):

        resnet = inputs
        in_channels = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(scope, default_name=name, values=[inputs],
                               reuse=reuse) as scope:

            for i in range(nb_blocks):
                identity = resnet
                if not downsample:
                    downsample_strides = 1

                if batch_norm:
                    resnet = self.bn("bn0",resnet)
                resnet = tf.nn.relu(resnet)

                resnet = self.conv2d(str(i)+"0",resnet, bottleneck_size, 1, downsample_strides, 'VALID')

                if batch_norm:
                    resnet = self.bn("bn0",resnet)
                resnet = tf.nn.relu(resnet)

                resnet = self.conv2d(str(i)+"1",resnet, bottleneck_size, 3, 1, 'SAME')

                resnet = self.conv2d(str(i)+"2",resnet, out_channels, 1, 1, 'VALID')
                resnet = tf.nn.relu(resnet)

                if downsample_strides > 1:
                    identity = self.avg_pool("avg_pool", identity, downsample_strides, downsample_strides)


                if in_channels != out_channels:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,[[0, 0], [0, 0], [0, 0], [ch, ch]])
                    in_channels = out_channels

                    resnet = resnet + identity
                    resnet = tf.nn.relu(resnet)

        return resnet

    def resnet_block(self, layer_name, inputs, layer_number, bottleneck_size, out_channels, downsample=False,downsample_strides=2):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            resnet = inputs
            in_channels = inputs.get_shape()[-1]
            for i in range(layer_number):
                identity = resnet
                if not downsample:
                    downsample_strides = 1

                resnet = self.conv2d(str(i)+"/1", resnet, bottleneck_size, 1, downsample_strides, 'VALID')

                resnet = self.conv2d(str(i)+"/2", resnet, bottleneck_size, 3, 1, 'SAME')

                resnet = self.conv2d(str(i) + "/3", resnet, out_channels, 1, 1, 'VALID')

                if downsample_strides >1:
                    identity = self.avg_pool(str(i)+"avg_pool", identity, downsample_strides, downsample_strides)
                    #NHWC do not change NHW so the pad [0,0][0,0][0,0]

                if in_channels != out_channels:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity,[[0,0],[0,0],[0,0],[ch,ch]])
                    in_channels = out_channels

                resnet = resnet + identity
        return resnet



    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]
        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]
        self.block_num = param["block_num"]
        self.class_num = param["class_num"]

        self.build_model()

        # 定义交叉熵损失函数
        self.global_step = tf.placeholder(dtype=tf.int32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))

        optimizer = param["optimizer"]
        base_learning_rate = param["learning_rate"]

        self.decay_steps = param["decay_steps"]
        self.decay_rate = param["decay_rate"]

        learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())


    def get_batch(self, feed_data):
        X = feed_data["inputs"]
        Y = feed_data["labels"]
        totalbatch = int(len(X) / self.batch_size) + 1

        if ( len(X) / self.batch_size * self.batch_size == len(X)):
            totalbatch = totalbatch - 1

        for i in range(0, totalbatch):
            startindex = i * self.batch_size
            endindex = (i + 1) * self.batch_size
            batch_xs = X[startindex:endindex]
            batch_ys = Y[startindex:endindex]

            yield {"batch_xs": batch_xs, "batch_ys": batch_ys}


    def train(self, feed_data):

        trainstep = int(0)
        for epoch in range(0, self.num_epochs):
            avg_cost = 0.0

            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs: batch["batch_xs"],
                    self.labels: batch["batch_ys"],
                    self.global_step: trainstep,
                }
                _, loss, train_accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                totalaccuracy += train_accuracy * len(batch["batch_xs"])
                avg_cost += loss
                trainstep = trainstep + 1
            totalaccuracy /= len(feed_data["inputs"])
            avg_cost /= len(feed_data["inputs"])
            print("accuracy:" + "\t" + str(totalaccuracy) + "\t" + "loss:" + "\t" + str(avg_cost))



    def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)


    def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)


    def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)


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


    def evaluate(self, feed_data):
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"]
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            totalaccuracy += acc * len(batch["batch_xs"])
            avg_loss += loss
        avg_loss /= totallen
        totalaccuracy /= len(feed_data['inputs'])

        res = {"accuracy": totalaccuracy, "loss": avg_loss}
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
