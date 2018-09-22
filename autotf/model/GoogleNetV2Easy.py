#-*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from util import *

import tensorflow as tf
import pickle

class GoogleNetV2(BaseModel):
    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25,
    }

    def __init__(self,classnum):
        self.class_num = classnum
        self.sess = tf.Session()
        self.summary = []
        self.scope = {}
    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])


        #227*227*3
        self.conv1_7x7_s2 = self.conv2d_bn('conv1_7x7_s2', self.inputs, 64, 7, 2)
        self.pool1_3x3_s2 = self.max_pool('pool1_3x3_s2', self.conv1_7x7_s2, 3, 2)
        #56*56*64


        self.conv2_3x3_reduce = self.conv2d_bn('conv2_3x3_reduce', self.pool1_3x3_s2, 64, 1, 1)
        self.conv2_3x3 = self.conv2d_bn('conv2_3x3', self.conv2_3x3_reduce, 192, 3, 1)

        self.pool2_3x3_s2 = self.max_pool('pool2_3x3_s2', self.conv2_3x3, 3, 2)

        #28*28*192

        self.inception_3a_output = self.InceptionV2("inception_3a",self.pool2_3x3_s2,64,64,64,64,96,32,"avg")
        #28*28*256

        self.inception_3b_output = self.InceptionV2("inception_3b",self.inception_3a_output,64,64,96,64,96,64,"avg")

        #64 + 96 + 96 + 64 = 320
        #28*28*320

        self.inception_3c_output =  self.InceptionV2("inception_3c",self.inception_3a_output,0,128,160,64,96,-1)

        #320 + 96 + 160 + 0 =576
        #14*14*576

        self.inception_4a_output =  self.InceptionV2("inception_4a",self.inception_3a_output,224,64,96,96,128,128,"avg")

        # 224+96+128+128 = 576
        #14*14*576
        self.inception_4b_output =  self.InceptionV2("inception_4b",self.inception_4a_output,192,96,128,96,128,128,"avg")

        # 192 +  128 + 128+128 = 576
        # 14*14*576
        self.inception_4c_output =  self.InceptionV2("inception_4c",self.inception_4b_output,160,128,160,128,160,128,"avg")

        #160+160+160+128 = 608
        #14*14*608
        self.inception_4d_output =  self.InceptionV2("inception_4d",self.inception_4b_output,96,128,192,160,192,128,"avg")

        #96+192+192+128=608
        #14*14*608
        self.inception_4e_output =  self.InceptionV2("inception_4e",self.inception_4b_output,0,128,192,192,256,-1)

        #192 + 256 + 608=1056
        #7*7*1056

        self.inception_5a_output =  self.InceptionV2("inception_5a",self.inception_4e_output,352,192,320,160,224,128,'avg')

        # 352+320+ 224+128 = 1024
        # 7*7*1024

        self.inception_5b_output =  self.InceptionV2("inception_5b",self.inception_4e_output,352,192,320,160,224,128,'max')

        # 352+320+224+128 = 1024
        # 7*7*1024

        self.pool5_7x7_s1 = self.avg_pool('pool5_7x7_s1', self.inception_5b_output, 7, 1)

        self.pred = self.fc('loss3_classifier', self.pool5_7x7_s1, out_nodes=self.class_num)

    def InceptionV2(self,layer_name,inputs,size1x1,size3x3reduce,size3x3,size3x3dreduce,sized3x3,sizepool,str="avg"):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            i1x1 = self.conv2d_bn('1x1', inputs, size1x1, 1, 1)

            i3x3_reduce = self.conv2d_bn('3x3_reduce', i1x1, size3x3reduce, 1,1)
            i3x3 = self.conv2d_bn('i3x3', i3x3_reduce, size3x3, 3, 1)

            id3x3_reduce = self.conv2d_bn('d_3x3_reduce', i3x3, size3x3dreduce,
                                                            1, 1)
            id3x3_1 = self.conv2d_bn('d_3x3_1', id3x3_reduce, sized3x3, 3,
                                                       1)
            id3x3_2 = self.conv2d_bn('d_3x3_2', id3x3_1, sized3x3, 3, 1)

            if size1x1==0:
                pool = self.max_pool('pool', id3x3_2, 3, 1)
                output = self.concatthreee('output', [i1x1, i3x3,id3x3_2,pool])
            else:
                if (str=="avg"):
                    pool = self.avg_pool('pool', id3x3_2, 3, 1)
                else:
                    pool = self.max_pool('pool', id3x3_2, 3, 1)
                pool_proj = self.conv2d_bn('pool_proj', pool, sizepool, 1, 1)
                output = self.concat('output', [i1x1, i3x3,id3x3_2,pool_proj])
            return output

    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()

        # 定义交叉熵损失函数
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))

        optimizer = param["optimizer"]
        learning_rate = param["learning_rate"]
        self.optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]
        print(self.num_epochs)

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
        for epoch in range(0,self.num_epochs):
            avg_cost = 0.0

            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                }
                _, loss, train_accuracy = self.sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
                totalaccuracy +=  train_accuracy*len(batch["batch_xs"])
                avg_cost +=  loss
                trainstep = trainstep + 1
            totalaccuracy /= len(feed_data["inputs"])
            print("train_step"+"\t"+str(trainstep)+"\t"+"epoch:"+"\t"+str(epoch+1)+"\t"+"accuracy:"+"\t"+str(totalaccuracy)+"\t"+"loss:"+"\t"+str(avg_cost))

    def conv2d_bn(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
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
            inputs = tf.layers.batch_normalization(inputs)
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs
    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
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

    def concatthreee(self,layer_name,inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([three_by_three, five_by_five, pooling], axis=3)

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
                self.labels: batch["batch_ys"]
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
