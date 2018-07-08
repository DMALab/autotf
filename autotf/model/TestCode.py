from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import re
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.data_utils import to_categorical, pad_sequences
from logistic_regression import *
from AlexNet import *
from vgg16 import *
from GoogleNet import *
from GoogleNetV2 import *
from RandomForest import *
from LinearSvm import *
from KernelSvm import *
from Rnn import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# This code for testing logistic regression training
def TestLR():
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-2,
        "batch_size": 200,
        "num_epochs": 25
    }

    mnist = input_data.read_data_sets("./data/", one_hot=True)
    X, Y = mnist.train.next_batch(mnist.train.num_examples)
    print("mnist examples:" + str(mnist.train.num_examples))
    print("lenX:" + str(len(X)))

    m = LogisticRegression(784, 10)
    m.set_parameter(params)
    inputdata = {"inputs": X, "labels": Y}
    m.train(inputdata)
    m.model_save("./ModelSavePath/logisstic_regression.ckpt")

    m.model_load("./ModelSavePath/logisstic_regression.ckpt")

    dic = m.evaluate(inputdata)
    print("Evaluate:" + str(dic))

#This Code for testing AlexNet
def TestAlexNet():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    X, Y = mnist.train.next_batch(mnist.train.num_examples)
    print("mnist examples:" + str(mnist.train.num_examples))
    print("lenX:" + str(len(X)))
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-2,
        "batch_size": 100,
        "num_epochs": 10,
        "keep_prob": 0.75
    }
    m = AlexNet(784, 10)
    m.set_parameter(params)
    print(mnist.train.num_examples)

    inputdata = {"inputs": X, "labels": Y}
    m.train(inputdata)
    m.model_save("./ModelSavePath/AlexNet.ckpt")
    m.model_load("./ModelSavePath/AlexNet.ckpt")

    dic = m.evaluate(inputdata)
    print("Evaluate:" + str(dic))

#This Code for testing vgg16
def TestVgg16():
    def GetInput():
        pkl_file = open('flower17/224X.pkl', 'rb')
        X = pickle.load(pkl_file)
        print(X.shape)

        pkl_file = open('flower17/224Y.pkl', 'rb')
        Y = pickle.load(pkl_file)
        print(Y.shape)
        return X, Y

    def GetValidation(X, Y, splitsize=0.1):
        totallen = int(len(X))
        endindex = int(totallen * (1 - splitsize))

        TrainX = X
        TrainY = Y
        ValidX = X
        ValidY = Y
        # ValidX = X[endindex:]
        # ValidY = Y[endindex:]
        return TrainX, TrainY, ValidX, ValidY

    X, Y = GetInput()
    m = Vgg16(17)
    np.set_printoptions(threshold='nan')
    TrainX, TrainY, ValidX, ValidY = GetValidation(X, Y, 0.1)

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 10,
        "keep_prob": 0.75
    }

    feed_data = {"inputs": TrainX, "labels": TrainY, "ValidX": ValidX, "ValidY": ValidY}
    m.set_parameter(params)
    m.train(feed_data)

    m.model_save("./ModelSavePath/vgg16.ckpt")
    m.model_load("./ModelSavePath/vgg16.ckpt")

    dic = m.evaluate(feed_data)
    print("Evaluate:" + str(dic))

#This Code for testing googlenet
def TestGoogleV1():
    def GetInput():
        pkl_file = open('flower17/X.pkl', 'rb')
        X = pickle.load(pkl_file)
        print(X.shape)

        pkl_file = open('flower17/Y.pkl', 'rb')
        Y = pickle.load(pkl_file)
        print(Y.shape)
        return X, Y

    X, Y = GetInput()
    m = GoogleNet(17)

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "num_epochs": 10,
    }

    feed_data = {"inputs": X, "labels": Y}
    m.set_parameter(params)
    m.train(feed_data)

    m.model_save("./ModelSavePath/GoogLeNet.ckpt")
    m.model_load("./ModelSavePath/GoogLeNet.ckpt")

    dic = m.evaluate(feed_data)
    print("Evaluate:" + str(dic))

def TestGoogleV2():
    def GetInput():
        pkl_file = open('flower17/X.pkl', 'rb')
        X = pickle.load(pkl_file)
        print(X.shape)

        pkl_file = open('flower17/Y.pkl', 'rb')
        Y = pickle.load(pkl_file)
        print(Y.shape)
        return X,Y
    X,Y = GetInput()
    m = GoogleNetV2(17)

    params = {
            "loss" : "square_loss",
            "metrics" : ["loss"],
            "optimizer" : "sgd",
            "learning_rate" : 1e-3,
            "batch_size" : 32,
            "num_epochs" : 100,
        }

    feed_data = {"inputs":X,"labels":Y}
    m.set_parameter(params)
    m.train(feed_data)


    m.model_save("./ModelSavePath/GoogLeNetV2.ckpt")
    m.model_load("./ModelSavePath/GoogLeNetV2.ckpt")

    dic = m.evaluate(feed_data)
    print("Evaluate:"+str(dic))

def TestRandomForest():
    params = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "batch_size" : 200,
        "num_epochs" : 1,
        "num_trees" : 10,
        "max_nodes":1000
    }

    mnist = input_data.read_data_sets("./data/", one_hot=False)

    m = RandomForest(784,10)
    m.set_parameter(params)

    X,Y = mnist.train.next_batch(mnist.train.num_examples)

    inputdata = {"inputs":X,"labels":Y}
    m.train(inputdata)
    m.model_save("./ModelSavePath/RandomForest.ckpt")
    m.model_load("./ModelSavePath/RandomForest.ckpt")

    dic = m.evaluate(inputdata)
    print("Evaluate:"+str(dic))

def TestLinearSvm():
    params = {
        "metrics" : [],
        "batch_size" : 256,
        "class_num":2,
        "num_epochs": 25,
    }

    mnist = input_data.read_data_sets("./data/", one_hot=False)
    X, Y = mnist.train.next_batch(mnist.train.num_examples)

    ids = np.where((Y == 4) | (Y == 9))
    images = X[ids]
    labels = Y[ids]
    labels = labels == 4
    # mnist 4 => label 1 and mnist 9=> label 0

    feed_data = {"inputs":images.copy(),"labels":labels.copy()}
    m = LinearSVM(784)
    m.set_parameter(params)
    m.train(feed_data)

    pre = m.predict(feed_data)
    idx = 0
    for i in range(0,len(labels)):
        if (labels[i] == pre[i]):
            idx = idx + 1

    print(idx)
    print("auc"+str(float(idx)/len(labels)))

    #m.model_save("./ModelSavePath/Svm.ckpt")
    #m.model_load("./ModelSavePath/Svm.ckpt")
    dic = m.evaluate(feed_data)
    print("Evaluate:" + str(dic))
def TestKernelSvm():

    params = {
        "metrics" : [],
        "batch_size" : 256,
        "class_num":2,
        "num_epochs": 25,
        "learning_rate": 0.3,
        "l2":0.001,
        "stddev":5.0,
        "out_dim":2000,
    }

    mnist = input_data.read_data_sets("./data/", one_hot=False)
    X, Y = mnist.train.next_batch(mnist.train.num_examples)

    ids = np.where((Y == 4) | (Y == 9))
    images = X[ids]
    labels = Y[ids]
    labels = labels == 4
    # mnist 4 => label 1 and mnist 9=> label 0

    feed_data = {"inputs":images.copy(),"labels":labels.copy()}
    m = KernelSVM(784)
    m.set_parameter(params)
    m.train(feed_data)

    pre = m.predict(feed_data)
    idx = 0
    for i in range(0,len(labels)):
        if (labels[i] == pre[i]):
            idx = idx + 1

    print(idx)
    print("auc"+str(float(idx)/len(labels)))

    #m.model_save("./ModelSavePath/KernelSvm.ckpt")
    #m.model_load("./ModelSavePath/KernelSvm.ckpt")
    dic = m.evaluate(feed_data)
    print("Evaluate:" + str(dic))

def TestRnn():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"RNNCell",
        "IsBidirection": False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic = m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/RnnModel.ckpt")
    m.model_load("/home/share/model/RnnModel.ckpt")
    predict = m.predict(test_feed_data)

def TestLstm():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"LSTMCell",
        "IsBidirection": False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/LstmModel.ckpt")
    m.model_load("/home/share/model/LstmModel.ckpt")
    predict = m.predict(test_feed_data)

def TestGRU():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"GRUCell",
        "IsBidirection":False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/GRUModel.ckpt")
    m.model_load("/home/share/model/GRUModel.ckpt")
    predict = m.predict(test_feed_data)

def TestBiRnn():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"RNNCell",
        "IsBidirection": True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic = m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/RnnModel.ckpt")
    m.model_load("/home/share/model/RnnModel.ckpt")
    predict = m.predict(test_feed_data)

def TestBiLstm():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"LSTMCell",
        "IsBidirection": True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/LstmModel.ckpt")
    m.model_load("/home/share/model/LstmModel.ckpt")
    predict = m.predict(test_feed_data)

def TestBiGRU():
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=100, value=0.)
    TestX = pad_sequences(TestX, maxlen=100, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)

    params = {
        "metrics": [],
        "batch_size": 64,
        "class_num": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":100,
        "embdding_dimension":128,
        "vocab_size":45647,
        "layer_num":3,
        "CellName":"GRUCell",
        "IsBidirection":True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": TrainX, "labels": TrainY}
    test_feed_data = {"inputs": TestX, "labels": TestY}
    m.train(feed_data, test_feed_data)
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/GRUModel.ckpt")
    m.model_load("/home/share/model/GRUModel.ckpt")
    predict = m.predict(test_feed_data)


TestRnn()
TestLstm()
TestGRU()
TestBiRnn()
TestBiLstm()
TestBiGRU()