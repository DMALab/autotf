from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random

class LinearSVM(BaseModel):
    default_param = {
        "metrics" : [],
        "batch_size" : 256,
        "class_num":2,
        "num_epochs": 25,
    }

    def __init__(self, feature_num, param=None):
        self.feature_num = feature_num
        self.class_num = None
        self.estimator = None
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.model_path = None
        self.num_eval_steps = 1
        if param:
            self.set_parameter(param)

    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]
        self.num_epochs = param["num_epochs"]
        self.class_num = param["class_num"]
        self.batch_size = param["batch_size"]

    def get_train_fn(self,capacity=10000,
                     min_after_dequeue=3000):
        def _input_fn():
            batchx, batchy = tf.train.shuffle_batch(
                tensors=[self.train_data,
                         self.train_label],
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                num_threads=4)
            features_map = {"images": batchx}
            return features_map, batchy
        return _input_fn

    def get_evaluate_fn(self,capacity=10000,
                     min_after_dequeue=3000):
        def _input_fn():
            batchx, batchy = tf.train.shuffle_batch(
                tensors=[self.eval_data,
                         self.eval_label],
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                num_threads=4)
            features_map = {"images": batchx}
            return features_map, batchy
        return _input_fn
    def get_predict_fn(self,capacity=10000):
        def _input_fn():
            features_map = {"images": tf.convert_to_tensor(self.predict_data)}
            return features_map
        return _input_fn

    def train(self, feed_data):
        self.train_data = feed_data["inputs"]
        self.train_label = feed_data["labels"]
        image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
        self.estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=self.class_num)
        train_input_fn = self.get_train_fn()
        self.estimator.fit(input_fn=train_input_fn, steps=self.num_epochs)

    def evaluate(self, feed_data):
        self.eval_data = feed_data["inputs"]
        self.eval_label = feed_data["labels"]
        eval_input_fn = self.get_evaluate_fn()
        eval_metrics = self.estimator.evaluate(input_fn=eval_input_fn, steps=1)
        return eval_metrics

    def predict(self, feed_data):
        self.predict_data = feed_data["inputs"]
        predict_input_fn = self.get_predict_fn()
        gen = self.estimator.predict(input_fn=predict_input_fn)
        result = [x for x in gen]
        return np.array(result)


    def model_load(self,path):
        return
    def model_save(self,path):
        return
