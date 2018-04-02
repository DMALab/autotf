import numpy as np

from collections import OrderedDict

from autotf.selector.base_selector import BaseSelector
from autotf.tuner.fmin import bayesian_optimization

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class AccurateSelector(BaseSelector):

    _LOGISTIC_OFFSET = 1
    _LOGISTIC_NUM = 2

    _LOGISTIC_PARAMS = OrderedDict({
        "penalty": ["l1", "l2"],
        "C": [0.1, 10.0]
        }
    )
    _SVM_OFFSET = 3
    _SVM_NUM = 4
    _SVM_PARAMS = OrderedDict({
        "C": [0.1, 10.0],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "degree": [0, 10],
        "coef0": [0.0, 1.0]
    })

    _KNN_OFFSET = 7
    _KNN_NUM = 1
    _KNN_PARAMS = OrderedDict({
        "n_neighbors": [3, 10]
    })

    PARAMS_SPACE = [_LOGISTIC_PARAMS, _SVM_PARAMS, _KNN_PARAMS]

    def select_model(self, X, y, metric=None):
        """
        Find the best model with its hyperparameters from the autotf's model zool
        """
        self.X = X
        self.y = y
        if self.learners is None:
            self.learners = BaseSelector.ALL_LEARNERS

        lower, upper = self._get_bounds()

        result = bayesian_optimization(self.cost_func, lower=lower, upper=upper, num_iterations=10)

        return result["x_opt"], result["f_opt"]

    def show_models(self, params, accu):
        pass
    #     model_type = params[0]
    #     configuration = {}
    #     if model_type = 0:
    #         configuration["classifier"] = "logis"
    #         configuration[]

    def cost_func(self, args):
        model_type = int(args[0])

        if model_type == 0:
            return self.logistic_fun(args[self._LOGISTIC_OFFSET: self._LOGISTIC_OFFSET + self._LOGISTIC_NUM])
        elif model_type == 1:
            return self.svm_fun(args[self._SVM_OFFSET: self._SVM_OFFSET + self._SVM_NUM])
        elif model_type == 2:
            return self.knn_fun(args[self._KNN_OFFSET: self._KNN_OFFSET + self._KNN_NUM])

    def logistic_fun(self, args):
        configuration = dict()
        configuration['clf'] = "LogisticRegression"

        penalty = args[0]
        penalty = self._LOGISTIC_PARAMS["penalty"][int(penalty)]
        configuration['penalty'] = penalty

        C = args[1]
        configuration["C"] = C

        clf = LogisticRegression(penalty=penalty, C=C)

        performance = np.average(cross_val_score(clf, X=self.X, y=self.y, cv=10))
        configuration["performance"] = performance

        print(configuration)

        return 1.0 - performance

    def svm_fun(self, args):
        configuration = dict()
        configuration['clf'] = 'SVC'

        C = args[0]
        configuration['C'] = C

        kernel = args[1]
        kernel = self._SVM_PARAMS["kernel"][int(kernel)]
        configuration['kernel'] = kernel

        degree = int(args[2])
        configuration['degree'] = degree

        coef0 = args[3]
        configuration['coef0'] = coef0

        clf = SVC(C=C, kernel=kernel, degree=degree, coef0=coef0)

        performance = np.average(cross_val_score(clf, X=self.X, y=self.y, cv=10))
        configuration['performance'] = performance

        print(configuration)
        return 1.0 - performance

    def knn_fun(self, args):
        configuration = dict()
        configuration['clf'] = 'knn'

        n_neighbors = int(args[0])
        configuration['n_neighbors'] = n_neighbors

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        performance = np.average(cross_val_score(clf, X=self.X, y=self.y, cv=10))
        configuration['performance'] = performance

        print(configuration)
        return 1 - performance

    def _get_bounds(self):
        lower_bound = []
        upper_bound = []

        # model type bound
        lower_bound.append(0)
        upper_bound.append(len(self.learners))

        for params in self.PARAMS_SPACE:
            for name in params.keys():

                if isinstance(params[name][0], float):
                    lower_bound.append(params[name][0])
                    upper_bound.append(params[name][1])

                elif isinstance(params[name][0], int):
                    lower_bound.append(params[name][0])
                    upper_bound.append(params[name][-1] + 1)
                else:
                    lower_bound.append(0)
                    upper_bound.append(len(params[name]))

        return np.array(lower_bound), np.array(upper_bound)
