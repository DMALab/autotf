from __future__ import division, print_function, absolute_import
import tflearn
import numpy
import copy
import time

X=numpy.random.rand(10,10).tolist()
#2 output features per instance
Y=numpy.random.rand(10,2).tolist()

class Model:
    paramName = []
    params = {}
    TrainResult = {}
    def train(self,X,Y):
        return

    def test(self,X):
        return

    def SetParams(self,InputParams):
        return

    def BuildModel(self):
        return

    def Default_param(self):
        return

    def model_load(self,path):
        return

    def model_save(self,path):
        return

class Regression(Model):

    def BuildModel(self):
        if self.params is not None:
            input_ = tflearn.input_data(shape=[None, 10])
            r1 = tflearn.fully_connected(input_, 10)
            r1 = tflearn.fully_connected(r1, 2)
            r1 = tflearn.regression(r1, optimizer=params['optimizer'], loss=params['loss'],
                                    metric='R2', learning_rate=params['learning_rate'])
            self.model = tflearn.DNN(r1)
        else:
            input_ = tflearn.input_data(shape=[None, 10])
            r1 = tflearn.fully_connected(input_, 10)
            r1 = tflearn.fully_connected(r1, 2)
            r1 = tflearn.regression(r1, optimizer='sgd', loss='mean_square',
                                    metric='R2', learning_rate=0.01)
            self.model = tflearn.DNN(r1)

    def __init__(self,InputParams):
        self.params = copy.deepcopy(InputParams)
        self.paramName = []
        for item in InputParams:
            self.paramName.append(item)
        if self.params is not None:
            input_ = tflearn.input_data(shape=[None, 10])
            r1 = tflearn.fully_connected(input_, 10)
            r1 = tflearn.fully_connected(r1, 2)
            r1 = tflearn.regression(r1, optimizer=params['optimizer'], loss=params['loss'],
                                    metric='R2', learning_rate=params['learning_rate'])
            self.model = tflearn.DNN(r1)
        else:
            input_ = tflearn.input_data(shape=[None, 10])
            r1 = tflearn.fully_connected(input_, 10)
            r1 = tflearn.fully_connected(r1, 2)
            r1 = tflearn.regression(r1, optimizer='sgd', loss='mean_square',
                                    metric='R2', learning_rate=0.01)
            self.model = tflearn.DNN(r1)
        return

    def SetParams(self,InputParams):
        self.params = copy.deepcopy(InputParams)
        self.paramName = []
        for item in InputParams:
            self.paramName.append(item)
        BuildModel()
        return

    def train(self,X,Y):
        timestart = time.time()
        self.model.fit(X,Y,n_epoch=params['n_epoch'],show_metric=False)
        timeend = time.time()
        traintime = timeend - timestart
        self.TrainResult['traintime']= traintime
        return

    def test(self,X):
        res = self.model.predict(X)
        return res

    def model_load(self,path):
        self.model.model_load(path)
        return

    def model_save(self,path):
        self.model.model_save(path)
        return

    def evaluate(self,EvaDic,X,Y):
        # need to get loss accuracy...
        timestart = time.time()
        res = self.model.predict(X)
        timeend = time.time()
        testtime = timeend-timestart
        loss = 0.0
        accuracy = 0.0
        res = {}
        if 'loss' in EvaDic:
            res['loss'] = loss
        if 'accuarcy' in EvaDic:
            res['loss'] = accuracy
        if 'traintime' in EvaDic:
            res['traintime'] = self.TrainResult['traintime']
        if 'testtime' in EvaDic:
            res['testtime'] = testtime
        return res

params = {'optimizer':'sgd','loss':'mean_square','learning_rate':0.01,'n_epoch':10}
reg = Regression(params)
#k.SetParams(params)

reg.train(X,Y)
testinstance=numpy.random.rand(1,10).tolist()
testres = reg.test(testinstance)
EvaDic = {'loss','accuracy','traintime','testtime'}
res  = reg.evaluate(EvaDic,testinstance,-1)
print(res)



