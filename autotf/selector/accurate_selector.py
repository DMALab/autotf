import numpy as np

from autotf.selector.base_selector import BaseSelector

from autotf.selector.funcs import *

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition


FOUT = open('./train_log.txt', 'w')
ALL_LEARNERS = ["logistic", "alexnet", "vgg16", "googlenet", "randomforest"]
ML_LEARNERS = ["randomforest", "linearsvm", "kernelsvm"]
CNN_LEARNERS = ["alexnet", "vgg16", "googlenet"]
RNN_LEARNERS = ["lstm"]
REGRESSION_LEARNERS = ["lr"]


def accuracy_score(y_true, y_pred):
    n = len(y_true)
    return np.sum(y_true == y_pred) / n


class AccurateSelector:

    def __init__(self, task_type):
        # super().__init__(args, kwargs)

        self.task_type = task_type
        self.configuration_space = None
        self.feature_num = None
        self.class_num = None

        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

    def select_model(self, X, y, feature_num, class_num, metric=None):
        """
        Find the best model with its hyperparameters from the autotf's model zool
        """
        if len(X) != len(y):
            raise ValueError("the input list of X and y must have same length!!")
        self.X = X
        self.y = y

        self.x_train = X[0]
        self.x_valid = X[1]

        self.y_train = y[0]
        self.y_valid = y[1]

        if len(X) == 3:
            self.x_test = X[2]
            self.y_test = y[2]

        self.feature_num = feature_num
        self.class_num = class_num

        self.set_configuration_space()
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": 10,
                             "cs": self.configuration_space,
                             "deterministic": "true"})
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=self.cost_func)
        best_cfg = smac.optimize()
        history = smac.get_runhistory()

        configs = history.get_all_configs()
        print("--------------Selection Done!!!!--------------------")
        for config in configs:
            print(config)
            print(history.get_cost(config))
            print("------------------------------------")

        return best_cfg, history.get_cost(best_cfg)

    def set_configuration_space(self):
        self.configuration_space = ConfigurationSpace()
        print("task_type = ", self.task_type)
        if self.task_type == "common_classification":
            learner_list = ML_LEARNERS
        elif self.task_type == "image_classification":
            learner_list = CNN_LEARNERS
        elif self.task_type == "text_classification":
            learner_list = RNN_LEARNERS
        else:
            learner_list = REGRESSION_LEARNERS
        print("learner_list = ", learner_list)
        model_type = CategoricalHyperparameter("model_type", learner_list, default_value=learner_list[0])
        self.configuration_space.add_hyperparameter(model_type)
        if self.task_type == "common_classification":

            # self.set_logistic_space(model_type)
            self.set_randomforest_space(model_type)
            self.set_linearsvm_space(model_type)
            self.set_kernelsvm_space(model_type)

        if self.configuration_space == "common_regression":
            pass

        if self.task_type == "image_classification":
            self.set_alexnet_space(model_type)
            self.set_googlenet_space(model_type)
            self.set_vgg16_space(model_type)

        if self.task_type == "text_classification":
            self.set_lstm_space(model_type)

    def set_logistic_space(self, model_type):

        logistic_loss = CategoricalHyperparameter("logistic_loss", ["square_loss"], default_value="square_loss")
        logistic_optimizer = CategoricalHyperparameter("logistic_optimizer", ["sgd"], default_value="sgd")
        logistic_learning_rate = UniformFloatHyperparameter("logistic_learning_rate", 1e-4, 1e-3, default_value=5e-4)
        logistic_batch_size = UniformIntegerHyperparameter("logistic_batch_size", 10, 100, default_value=30)
        logistic_num_epochs = UniformIntegerHyperparameter("logistic_num_epochs", 10, 50, default_value=25)

        logistic_parameters = list()
        logistic_parameters.append(logistic_loss)
        logistic_parameters.append(logistic_optimizer)
        logistic_parameters.append(logistic_learning_rate)
        logistic_parameters.append(logistic_batch_size)
        logistic_parameters.append(logistic_num_epochs)
        self.configuration_space.add_hyperparameters(logistic_parameters)

        for parameter in logistic_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter, parent=model_type, values=["logistic"]))

    def set_alexnet_space(self, model_type):
        alexnet_loss = CategoricalHyperparameter("alexnet_loss", ["square_loss"], default_value="square_loss")
        alexnet_optimizer = CategoricalHyperparameter("alexnet_optimizer", ["sgd"], default_value="sgd")
        alexnet_learning_rate = UniformFloatHyperparameter("alexnet_learning_rate", 1e-4, 1e-2, default_value=5e-4)
        alexnet_batch_size = UniformIntegerHyperparameter("alexnet_batch_size", 10, 100, default_value=50)
        alexnet_num_epochs = UniformIntegerHyperparameter("alexnet_num_epochs", 10, 50, default_value=25)
        alexnet_keep_prob = UniformFloatHyperparameter("alexnet_keep_prob", 0.7, 0.8, default_value=0.75)

        alexnet_parameters = list()
        alexnet_parameters.append(alexnet_loss)
        alexnet_parameters.append(alexnet_optimizer)
        alexnet_parameters.append(alexnet_learning_rate)
        alexnet_parameters.append(alexnet_batch_size)
        alexnet_parameters.append(alexnet_num_epochs)
        alexnet_parameters.append(alexnet_keep_prob)
        self.configuration_space.add_hyperparameters(alexnet_parameters)

        for parameter in alexnet_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter, parent=model_type, values=["alexnet"]))

    def set_vgg16_space(self, model_type):
        vgg16_loss = CategoricalHyperparameter("vgg16_loss", ["square_loss"], default_value="square_loss")
        vgg16_optimizer = CategoricalHyperparameter("vgg16_optimizer", ["sgd"], default_value="sgd")
        vgg16_learning_rate = UniformFloatHyperparameter("vgg16_learning_rate", 1e-4, 1.0, default_value=1e-2)
        vgg16_batch_size = UniformIntegerHyperparameter("vgg16_batch_size", 10, 100, default_value=50)
        vgg16_num_epochs = UniformIntegerHyperparameter("vgg16_num_epochs", 10, 30, default_value=20)
        vgg16_keep_prob = UniformFloatHyperparameter("vgg16_keep_prob", 0.7, 0.8, default_value=0.75)

        vgg16_parameters = list()
        vgg16_parameters.append(vgg16_loss)
        vgg16_parameters.append(vgg16_optimizer)
        vgg16_parameters.append(vgg16_learning_rate)
        vgg16_parameters.append(vgg16_batch_size)
        vgg16_parameters.append(vgg16_num_epochs)
        vgg16_parameters.append(vgg16_keep_prob)
        self.configuration_space.add_hyperparameters(vgg16_parameters)

        for parameter in vgg16_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter, parent=model_type, values=["vgg16"]))

    def set_googlenet_space(self, model_type):
        googlenet_loss = CategoricalHyperparameter("googlenet_loss", ["square_loss"], default_value="square_loss")
        googlenet_optimizer = CategoricalHyperparameter("googlenet_optimizer", ["sgd"], default_value="sgd")
        googlenet_learning_rate = UniformFloatHyperparameter("googlenet_learning_rate", 1e-4, 1e-2, default_value=5e-4)
        googlenet_batch_size = UniformIntegerHyperparameter("googlenet_batch_size", 10, 100, default_value=50)
        googlenet_num_epochs = UniformIntegerHyperparameter("googlenet_num_epochs", 10, 50, default_value=30)

        googlenet_parameters = list()
        googlenet_parameters.append(googlenet_loss)
        googlenet_parameters.append(googlenet_optimizer)
        googlenet_parameters.append(googlenet_learning_rate)
        googlenet_parameters.append(googlenet_batch_size)
        googlenet_parameters.append(googlenet_num_epochs)
        self.configuration_space.add_hyperparameters(googlenet_parameters)

        for parameter in googlenet_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter, parent=model_type, values=["googlenet"]))

    def set_randomforest_space(self, model_type):
        randomforest_batch_size = UniformIntegerHyperparameter("randomforest_batch_size", 10, 100, default_value=50)
        randomforest_num_epochs = UniformIntegerHyperparameter("randomforest_num_epochs", 10, 30, default_value=20)
        randomforest_num_trees = UniformIntegerHyperparameter("randomforest_num_trees", 10, 100, default_value=50)

        randomforest_parameters = list()
        randomforest_parameters.append(randomforest_batch_size)
        randomforest_parameters.append(randomforest_num_epochs)
        randomforest_parameters.append(randomforest_num_trees)
        self.configuration_space.add_hyperparameters(randomforest_parameters)

        for parameter in randomforest_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter,
                                                               parent=model_type,
                                                               values=["randomforest"]))

    def set_linearsvm_space(self, model_type):
        linearsvm_batch_size = UniformIntegerHyperparameter("linearsvm_batch_size", 200, 300, default_value=256)
        linearsvm_num_epochs = UniformIntegerHyperparameter("linearsvm_num_epochs", 10, 40, default_value=25)

        linearsvm_parameters = list()
        linearsvm_parameters.append(linearsvm_batch_size)
        linearsvm_parameters.append(linearsvm_num_epochs)
        self.configuration_space.add_hyperparameters(linearsvm_parameters)

        for parameter in linearsvm_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter,
                                                               parent=model_type,
                                                               values=["linearsvm"]))

    def set_kernelsvm_space(self, model_type):
        kernelsvm_batch_size = UniformIntegerHyperparameter("kernelsvm_batch_size", 200, 300, default_value=256)
        kernelsvm_num_epochs = UniformIntegerHyperparameter("kernelsvm_num_epochs", 10, 40, default_value=25)
        kernelsvm_learning_rate = UniformFloatHyperparameter("kernelsvm_learning_rate", 0.01, 1.0, default_value=0.1)

        kernelsvm_parameters = list()
        kernelsvm_parameters.append(kernelsvm_batch_size)
        kernelsvm_parameters.append(kernelsvm_num_epochs)
        kernelsvm_parameters.append(kernelsvm_learning_rate)

        self.configuration_space.add_hyperparameters(kernelsvm_parameters)

        for parameter in kernelsvm_parameters:
            self.configuration_space.add_condition(InCondition(child=parameter,
                                                               parent=model_type,
                                                               values=["kernelsvm"]))

    def set_lstm_space(self, model_type):
        lstm_batch_size = UniformIntegerHyperparameter("lstm_batch_size", 10, 50, default_value=20)
        lstm_num_epochs = UniformIntegerHyperparameter("lstm_num_epochs", 10, 50, default_value=20)

    def cost_func(self, args):

        model_type = args["model_type"]
        if model_type == "logistic":
            return logistic_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                 [self.y_train, self.y_valid, self.y_test],
                                 self.feature_num, self.class_num)
        if model_type == "alexnet":
            return alexnet_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                [self.y_train, self.y_valid, self.y_test],
                                self.feature_num, self.class_num)
        if model_type == "vgg16":
            return vgg16_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                              [self.y_train, self.y_valid, self.y_test],
                              self.feature_num, self.class_num)
        if model_type == "googlenet":
            return googlenet_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                  [self.y_train, self.y_valid, self.y_test],
                                  self.feature_num, self.class_num)
        if model_type == "randomforest":
            return randomforest_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                     [self.y_train, self.y_valid, self.y_test],
                                     self.feature_num, self.class_num)
        if model_type == "linearsvm":
            return linearsvm_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                  [self.y_train, self.y_valid, self.y_test],
                                  self.feature_num, self.class_num)
        if model_type == "kernelsvm":
            return kernelsvm_func(args, FOUT, [self.x_train, self.x_valid, self.x_test],
                                  [self.y_train, self.y_valid, self.y_test],
                                  self.feature_num, self.class_num)
