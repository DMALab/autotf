from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.learn import learn_runner
from tensorflow.examples.tutorials.mnist import input_data
from base_model import BaseModel


class BoostTree(BaseModel):

    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "batch_size" : 60000,
        "num_epochs" : 25,
        "learning_rate":0.3,
        "depth": 4,
        "examples_per_layer":60000,
        "eval_batch_size":10000,
        "num_eval_steps":1,
        "num_trees":10,
        "L2":1.0,
        "vmodule":1,
        "class_num":10,
        "model_path":"/tmp/dir",
        "objective":"multiclass",
    }

    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.class_num = None
        self.model = None
        self.estimator = None
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.model_path = None
        self.num_eval_steps = 1

    def get_input_fn(self,capacity=10000,
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




    def _make_experiment_fn(self,output_dir):
        """Creates experiment for gradient boosted decision trees."""

        train_input_fn = self.get_input_fn()
        eval_input_fn = self.get_input_fn()

        return tf.contrib.learn.Experiment(
            estimator=self.estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=None,
            eval_steps=self.num_eval_steps,
            eval_metrics=None)


    def build_model(self):
        return

    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()

        self.learner_config = learner_pb2.LearnerConfig()

        self.learner_config.learning_rate_tuner.fixed.learning_rate = float(param['learning_rate'])

        self.learner_config.regularization.l1 = 0.0
        self.learner_config.regularization.l2 = float(param['L2']) / int(param['examples_per_layer'])
        self.learner_config.constraints.max_tree_depth = int(param['depth'])

        self.growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER
        self.learner_config.growing_mode = self.growing_mode
        self.run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
        self.model_path = param['model_path']
        self.class_num = int(param['class_num'])
        if param['objective'] is "multiclass":
            print("here")

            self.learner_config.num_classes = param['class_num']
            self.learner_config.multi_class_strategy = (
                learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

        # Create a TF Boosted trees estimator that can take in custom loss.
            self.estimator = GradientBoostedDecisionTreeClassifier(
                learner_config=self.learner_config,
                n_classes=int(self.class_num),
                examples_per_layer=int(param['examples_per_layer']),
                model_dir=self.model_path,
                num_trees=int(param['num_trees']),
                center_bias=False,
                config=self.run_config)
        else:
            pass

        self.batch_size = int(param["batch_size"])
        self.eval_batch_size = int(param['eval_batch_size'])
        self.num_epochs = param["num_epochs"]


    def train(self, feed_data):
        self.train_data = feed_data["inputs"]
        self.train_label = feed_data["labels"]
        learn_runner.run(
            experiment_fn=self._make_experiment_fn,
            output_dir=str("/tmp/mnist"),
            schedule="train_and_evaluate")

    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return


