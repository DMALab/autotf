#coding=utf-8
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

class AlexNet(BaseModel):

    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25,
        "keep_prob":0.75
    }

    def __init__(self, feature_num,classnum):
        self.feature_num = feature_num
        self.class_num = classnum
        self.model = None
        self.sess = tf.Session()

    def build_model(self):
        with tf.variable_scope("AlexNet"):
            # 训练数据
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.feature_num])
            # 训练标签数据
            self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])
            # dropout
            self.keep_prob = tf.placeholder(tf.float32)

            # 把inputs更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
            self.x = tf.reshape(self.inputs, [-1, 28, 28, 1])

            # 第一层卷积
            self.conv1_weights = tf.Variable(tf.random_normal([3, 3, 1, 64]))  # 卷积核大小为3*3, 当前层深度为1， 过滤器深度为64

            # 卷积
            self.conv1 = tf.nn.conv2d(self.x, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            self.conv1_biases = tf.Variable(tf.random_normal([64]))

            # 激活函数Relu去线性化
            self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_biases))

            # 最大池化
            self.pool1 = tf.nn.max_pool(self.relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充

            # 规范化
            self.norm1 = tf.nn.lrn(self.pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            self.norm1 = tf.nn.dropout(self.norm1, self.keep_prob)
            print(self.norm1.shape)  # 14*14*64

            # 第二层卷积
            self.conv2_weights = tf.Variable(tf.random_normal([3, 3, 64, 128]))  # 卷积核大小为3*3, 当前层深度为64， 过滤器深度为128
            self.conv2 = tf.nn.conv2d(self.norm1, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            self.conv2_biases = tf.Variable(tf.random_normal([128]))
            self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, self.conv2_biases))
            self.pool2 = tf.nn.max_pool(self.relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.norm2 = tf.nn.lrn(self.pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            self.norm2 = tf.nn.dropout(self.norm2, self.keep_prob)
            print(self.norm2.shape)  # 7*7*128

            # 第三层卷积
            self.conv3_weights = tf.Variable(tf.random_normal([3, 3, 128, 256]))  # 卷积核大小为3*3, 当前层深度为128， 过滤器深度为256
            self.conv3 = tf.nn.conv2d(self.norm2, self.conv3_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            self.conv3_biases = tf.Variable(tf.random_normal([256]))
            self.relu3 = tf.nn.relu(tf.nn.bias_add(self.conv3, self.conv3_biases))
            self.pool3 = tf.nn.max_pool(self.relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.norm3 = tf.nn.lrn(self.pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            self.norm3 = tf.nn.dropout(self.norm3, self.keep_prob)
            print(self.norm3.shape)  # 4*4*256

            # 全连接层 1
            self.fc1_weights = tf.Variable(tf.random_normal([4 * 4 * 256, 1024]))
            self.fc1_biases = tf.Variable(tf.random_normal([1024]))
            self.fc1 = tf.reshape(self.norm3, [-1, self.fc1_weights.get_shape().as_list()[0]])
            self.fc1 = tf.add(tf.matmul(self.fc1, self.fc1_weights), self.fc1_biases)
            self.fc1 = tf.nn.relu(self.fc1)

            # 全连接层 2
            self.fc2_weights = tf.Variable(tf.random_normal([1024, 1024]))
            self.fc2_biases = tf.Variable(tf.random_normal([1024]))
            self.fc2 = tf.reshape(self.fc1, [-1, self.fc2_weights.get_shape().as_list()[0]])
            self.fc2 = tf.add(tf.matmul(self.fc2, self.fc2_weights), self.fc2_biases)
            self.fc2 = tf.nn.relu(self.fc2)

            # 输出层
            self.out_weights = tf.Variable(tf.random_normal([1024, 10]))
            self.out_biases = tf.Variable(tf.random_normal([10]))
            self.pred = tf.add(tf.matmul(self.fc2, self.out_weights), self.out_biases)

    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        # 定义交叉熵损失函数
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))
        self.keep_prob_value = param["keep_prob"]


        #loss_fun = param["loss"]
        #self.loss = loss_fun(self.output, self.ground_truth)

        #metrics = [self.get_metric(metric) for metric in param["metrics"]]
        #self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = param["optimizer"]
        self.learning_rate = param["learning_rate"]

        # 选择优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        for i in range(0,int(feed_data.train.num_examples/self.batch_size)):
            batch_xs,batch_ys = feed_data.train.next_batch(self.batch_size)
            yield { "batch_xs" : batch_xs, "batch_ys" : batch_ys }

    def train(self, feed_data):
        self.sess.run(tf.global_variables_initializer())
        print("here")

        for epoch in range(self.num_epochs):
            avg_cost = 0.0
            i = int(0)
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                    self.keep_prob: self.keep_prob_value,
                }
                _, loss, train_accuracy = self.sess.run([self.train_op, self.cost,self.accuracy], feed_dict=feed_dict)
                i = i + 1
                if (i%100 == 0):
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                avg_cost +=  loss
            avg_cost /= int(feed_data.train.num_examples/self.batch_size)
            print(avg_cost)


    def evaluate(self, feed_data):

        return
params = {
    "loss" : "square_loss",
    "metrics" : ["loss"],
    "optimizer" : "Adam",
    "learning_rate" : 1e-2,
    "batch_size" : 100,
    "num_epochs" : 25
}

m = AlexNet(784,10)
m.set_parameter(params)
print(mnist.train.num_examples)
m.train(mnist)
