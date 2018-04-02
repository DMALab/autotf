#-*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from autotf.model.base_model import BaseModel
from autotf.model.helper import *
import tensorflow as tf

class SeqClassificationRNN(BaseModel):

    default_param = {
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "max_length" : 200,
        "vocab_size" : 10000,
        "cell" : "LSTM", # or "GRU"
        "embed_dim" : 64,
        "rnn_hidden_dim" : 64,
        "layer_num" : 3,
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25,
        "keep_prob":0.75
    }

    def __init__(self, feature_num, classnum):
        self.feature_num = feature_num
        self.class_num = classnum
        self.model = None
        self.sess = tf.Session()

    def build_model(self):
        with tf.variable_scope("SeqClassificationRNN"):
            # 训练数据
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.max_length])
            self.seqlen = tf.placeholder(tf.int32, shape=[None])
            # 训练标签数据
            self.labels = tf.placeholder(tf.int32, shape=[None])
            onehot = tf.one_hot(self.labels, self.class_num)
            # dropout
            self.keep_prob = tf.placeholder(tf.float32)

            embedding_matrix = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_dim), stddev=0.01))

            embedding = tf.nn.embedding_lookup(embedding_matrix, self.inputs)

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.cell(self.hidden_dim, state_is_tuple=True) for _ in range(self.layer_num)])

            outputs, state = tf.nn.dynamic_rnn(cell, embedding, self.seqlen, dtype=tf.float32, swap_memory=True)

            pooling = tf.reduce_sum(outputs, 1) / self.seqlen
            dropout = tf.dropout(pooling, keep_prob=self.keep_prob)

            lr_W = tf.Variable(tf.truncated_normal((self.hidden_dim, self.class_num), stddev=0.1))
            lr_b = tf.Variable(tf.zeros((1, self.class_num)))

            pred = tf.matmul(dropout, lr_W) + lr_b

            # 定义交叉熵损失函数
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, onehot))

    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.max_length = param["max_length"]
        self.vocab_size = param["vocab_size"]
        self.embed_dim = param["embed_dim"]
        self.hidden_dim = param["rnn_hidden_dim"]
        self.layer_num = param["layer_num"]
        self.cell = {"LSTM" : tf.nn.rnn_cell.LSTMCell,
                     "GRU" : tf.nn.rnn_cell.GRUCell} [param["cell"]]

        self.build_model()
        self.keep_prob_value = param["keep_prob"]

        metrics = [self.get_metric(metric) for metric in param["metrics"]]
        self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = self.get_optimizer(param["optimizer"])
        learning_rate = param["learning_rate"]
        self.optimizer = optimizer(learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.labels)
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
                _, loss, train_accuracy = self.sess.run([self.train_op, self.cost, self.accuracy], feed_dict=feed_dict)
                i = i + 1
                if (i%100 == 0):
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                avg_cost +=  loss
            avg_cost /= int(feed_data.train.num_examples/self.batch_size)
            print(avg_cost)

    def evaluate(self, feed_data):
        return
