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
from GoogleNetV1Easy import *
from GoogleNetV2Easy import *
from GoogleNetV3 import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#This Code for testing googlenet
def GetData(width,height):
    def getimage(X):
        result = []
        for idx in range(0,len(X)):
            if idx % 10000 == 0:
                print("load data:\t" + str(idx) + "/" + str(len(inputx)))
            im = cv2.resize(X[idx],(width,height),interpolation=cv2.INTER_CUBIC)
            result.append(im)
        return result

    (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/share/cnndata/",one_hot=True)
    X = getimage(X)
    X_test = getimage(X_test)
    return X,Y,X_test,Y_test




def GetCifar10Batch(width,height,inputX):
    def getimage(inputx):
        result = []
        for idx in range(0, len(inputx)):
            im = cv2.resize(inputx[idx], (width, height), interpolation=cv2.INTER_CUBIC)
            result.append(im)
        return result
    X = getimage(inputX)
    return X

def TestVGG16():
    m = Vgg16(10)


    X, Y, X_test, Y_test = GetData(224,224)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_epochs": 100,
        "keep_prob": 0.8
    }
    feed_data = {"inputs": X, "labels": Y, "ValidX": X_test, "ValidY": Y_test}
    m.set_parameter(params)
    m.train(feed_data)
    m.model_save("./ModelSavePath/vgg16.ckpt")
    m.model_load("./ModelSavePath/vgg16.ckpt")
    dic = m.evaluate(feed_data)
    print("Evaluate:" + str(dic))
    return

def TestGoogleV1():
    m = GoogleNetV1(10)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 100,
    }
    X, Y, X_test, Y_test = GetData(227,227)
    feed_data = {"inputs": X, "labels": Y}
    test_feed_data = {"inputs":X_test,"labels":Y_test}
    m.set_parameter(params)

    time_start = time.time()
    m.train(feed_data)
    time_end =  time.time()
    time_delta = time_end - time_start
    print(time_delta/1000)

    m.model_save("/home/share/model/GoogLeNet.ckpt")
    m.model_load("/home/share/model/GoogLeNet.ckpt")

    dic = m.evaluate(test_feed_data)
    print("Evaluate:" + str(dic))

def TestGoogleV2():
    m = GoogleNetV2(10)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 100,
    }
    X, Y, X_test, Y_test = GetData(227, 227)
    feed_data = {"inputs": X, "labels": Y}
    test_feed_data = {"inputs":X_test,"labels":Y_test}
    m.set_parameter(params)

    time_start = time.time()
    m.train(feed_data)
    time_end =  time.time()

    time_delta = time_end - time_start
    print(time_delta/1000)

    m.model_save("/home/share/model/GoogLeNet.ckpt")
    m.model_load("/home/share/model/GoogLeNet.ckpt")

    dic = m.evaluate(test_feed_data)
    print("Evaluate:" + str(dic))

def TestGoogleV3():
    m = GoogleNetV3(10)
    (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/share/cnndata/", one_hot=True)

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 1,
    }

    m.set_parameter(params)
    time_start = time.time()
    for max_step in range(0,30000):
        index = np.random.choice(np.arange(len(X)),32,replace=False)

        curbatchx = X[index]
        curbatchy = Y[index]
        curbatchx= GetCifar10Batch(299, 299,curbatchx)
        feed_data = {"inputs": curbatchx, "labels": curbatchy}
        acc = m.train_batch(feed_data)
        if (max_step%300 == 0):
            print("step:"+str(max_step))
            print("accuracy"+str(acc))


    '''
    for batchnumber in range(0,5):
        print("batch:\t"+str(batchnumber))
        index = np.random.permutation(len(X))
        xinput = X[index]
        Ycur = Y[index]

        xinput = xinput[0:10000]
        Ycur = Ycur[0:10000]
        #xinput = X[10000*batchnumber:10000*(batchnumber+1)]
        #Ycur = Y[10000*batchnumber:10000*(batchnumber+1)]
        Xcur= GetCifar10Batch(299, 299,xinput)
        print(len(Xcur))
        print(len(Ycur))
    '''

    X_test = GetCifar10Batch(299,299,X_test)
    test_feed_data = {"inputs":X_test,"labels":Y_test}
    time_end = time.time()
    time_delta = time_end - time_start
    print(time_delta/1000)

    m.model_save("/home/share/model/GoogLeNetV3.ckpt")
    m.model_load("/home/share/model/GoogLeNetV3.ckpt")

    dic = m.evaluate(test_feed_data)
    print("Evaluate:" + str(dic))


#TestVGG16()
#TestGoogleV1()
#TestGoogleV2()
TestGoogleV3()


