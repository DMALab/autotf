from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import re
import cv2
import time
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.datasets import cifar10
from tflearn.data_utils import to_categorical, pad_sequences
from logistic_regression import *
from AlexNet import *
from vgg16 import *
from GoogleNet import *
from GoogleNetV2 import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#This Code for testing googlenet
def TestGoogleV1():
    m = GoogleNet(10)
    def getimage(X):
        result = []
        for idx in range(0,len(X)):
            if idx % 10000 == 0:
                print("load data:\t"+str(idx)+"/"+str(len(X)))
            im = cv2.resize(X[idx],(227,227),interpolation=cv2.INTER_CUBIC)
            result.append(im)
        return result

    (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/share/cnndata/",one_hot=True)
    X = getimage(X)
    X_test = getimage(X_test)

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 100,
    }

    feed_data = {"inputs": X, "labels": Y}
    test_feed_data = {"inputs":X_test,"labels":Y_test}
    m.set_parameter(params)
    time_start = time.time()
    m.train(feed_data)
    time_end =  time.time()
    print(time_end-time_start)


    m.model_save("/home/share/model/GoogLeNet.ckpt")
    m.model_load("/home/share/model/GoogLeNet.ckpt")

    dic = m.evaluate(test_feed_data)
    print("Evaluate:" + str(dic))
TestGoogleV1()


