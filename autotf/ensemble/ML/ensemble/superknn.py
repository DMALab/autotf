"""多进程且一次性predict,把randomsearch拆出来"""
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from ensemble import Stacking
import warnings
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import math
import numpy as np
import logging
import time
from multiprocessing import Pool, Queue

result = Queue()
logger = logging.getLogger('ensemble.superknn')

def find_distance(x,y,epsilon):
    alpha = 0
    for i in range(0,len(x)):
        if math.fabs(x[i]-y[i])<=epsilon:
            alpha+=1
    return alpha

def find_equal(x,y):
    alpha = 0
    for i in range(0, len(x)):
        if x[i] == y[i]:
            alpha += 1
    return alpha

class Superknn(BaseEstimator):
    """
    COBRA: A combined regression strategy.
    Based on the paper by Biau, Fischer, Guedj, Malley [2016].
    This is a pythonic implementation of the original COBRA code.

    Parameters
    ----------
    random_state: integer or a numpy.random.RandomState object.
        Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
        reproducibility of your experiments, for example.

    epsilon: float, optional
        Epsilon value described in the paper which determines which points are selected for the aggregate.
        Default value is determined by optimizing over a grid if test data is provided.
        If not, a mean of the possible distances is chosen.

    k: float, default 0.5
        k value described in the paper which determines how many points are selected in the optimal process

    alpha: int, optional
        alpha refers to the number of machines the prediction must be close to to be considered during aggregation.

    regression : boolean, default True
        If True - perform stacking for regression task,
        if False - perform stacking for classification task

    metric:callable, default None
        Evaluation metric (score function) which is used to calculate results of each split.

    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split

    models : default None
        used to fit and predict the data

    Attributes
    ----------
    machines_: A dictionary which maps machine names to the machine objects.
            The machine object must have a predict method for it to be used during aggregation.

    machine_predictions_: A dictionary which maps machine name to it's predictions over X_l
            This value is used to determine which points from y_l are used to aggregate.

    all_predictions_: numpy array with all the predictions, to be used for epsilon manipulation.

    """

    def __init__(self,X = None, y=None, regression=True, metric = None, shuffle=None,
                 random_state=None, epsilon=None, needs_proba=False,
                 alpha=None, models=None, folds=None, verbose=True, n_jobs =1):
        self.random_state = random_state
        self.epsilon = epsilon
        self.alpha = alpha
        self.models = models
        self.shuffle = shuffle
        self.verbose = verbose
        self.metric = metric
        self.n_jobs =n_jobs
        self.needs_proba = needs_proba
        if needs_proba and metric == 'accuracy':
            self.metric = log_loss
            warn_str = 'Task needs probability, so the metric is set to log_loss '
            warnings.warn(warn_str, UserWarning)
        self.folds = folds
        self.X_ = X
        self.y_ = y
        self.regression = regression


    def opmimal_parameters(self, X = None, y=None, eps_size=None,  grid_points=None):
        """
        Parameters
        ----------
        X: array-like, [n_samples, n_features]
           data which will be used to find the optimal parameters.
        y: array-like, [n_samples, n_features]
           data which will be used to find the optimal parameters.
        eps_size: float
           determines how many data are used in this process
        grid_points: int, optional
            If no epsilon value is passed, this parameter controls how many points on the grid to traverse.
        """

        _, X_opt, _, y_opt = train_test_split(X, y, test_size=eps_size, shuffle=self.shuffle,
                                             random_state=self.random_state)
        print("optimal data:", len(y_opt), "\n")
        if self.regression:
            print("Task: regression\n")
            print("---------Finding the optimal epsilon and alpha---------")
        else:
            if self.needs_proba:
                print("Task: classification\n")
                print("---------Finding the optimal epsilon and alpha---------")
            else:
                print("Task: classification\n")
                print("---------Finding the optimal  alpha---------")
        if self.epsilon is None and X_opt is not None:
            if self.shuffle:
                self.X_, self.y_ = shuffle(X_opt, y_opt, random_state=self.random_state)
            else:
                self.X_ = X_opt
                self.y_ = y_opt
            if self.needs_proba:
                arange = range(1, len(self.models)*len(set(self.y_)) + 1)
            else:
                arange = range(1, len(self.models) + 1)
            print('possible value of alpha:\n', arange, "\n")
            # get the candidate epsilons
            if self.regression:
                start_time = time.time()
                stack_model = Stacking(X_train=self.X_, y_train=self.y_, X_test=None, bagged_pred=False,
                                       regression=self.regression, metric=self.metric, n_folds=self.folds,
                                       shuffle=self.shuffle, random_state=0, verbose=0)
                stack_model.add(self.models)
                print('stacking took %fs!' % (time.time() - start_time))

                self.stack_X = stack_model.next_train
                a, size = sorted(self.stack_X.ravel()), len(self.stack_X.ravel())
                res = [a[i + 1] - a[i] for i in range(size) if i + 1 < size]
                emin = min(res)
                emax = max(a) - min(a)
                erange = np.linspace(emin, emax, grid_points)
                print('possible value of epsilon:\n', erange, "\n")
                tuned_parameters = {'epsilon': erange, 'alpha': arange}
                n_iter_search = int(math.sqrt(len(erange) * len(arange)))
                print("using random search for", n_iter_search, " candidates parameter settings\n")
            else:
                if self.needs_proba:
                    start_time = time.time()
                    stack_model = Stacking(X_train=self.X_, y_train=self.y_, X_test=None, bagged_pred=False,
                                           regression=self.regression, needs_proba=self.needs_proba,
                                           metric=self.metric, n_folds=self.folds,
                                           shuffle=True, random_state=0, verbose=0)
                    stack_model.add(self.models)
                    print('stacking took %fs!' % (time.time() - start_time))
                    self.stack_X = stack_model.next_train
                    a, size = sorted(self.stack_X.ravel()), len(self.stack_X.ravel())
                    res = [a[i + 1] - a[i] for i in range(size) if i + 1 < size]
                    emin = min(res)
                    emax = max(a) - min(a)
                    erange = np.linspace(emin, emax, grid_points)
                    print('possible value of epsilon:\n', erange, "\n")
                    tuned_parameters = {'epsilon': erange, 'alpha': arange}
                    n_iter_search = int(math.sqrt(len(erange) * len(arange)))
                    print("using random search for", n_iter_search, " candidates parameter settings\n")
                else:
                    tuned_parameters = {'alpha': arange}
                    print("using grid search for", len(arange), " candidates parameter settings\n")

            print('please wait...')
            if self.regression:
                clf = RandomizedSearchCV(Superknn(epsilon=self.epsilon, models=self.models,
                                                  regression=self.regression, metric=self.metric,
                                                  shuffle=self.shuffle, folds=self.folds,
                                                  n_jobs=1, verbose=False,
                                                  random_state=self.random_state),
                                         return_train_score=False,
                                         param_distributions=tuned_parameters, n_iter=n_iter_search,
                                         scoring="neg_mean_absolute_error")
            else:
                if self.needs_proba:
                    clf = RandomizedSearchCV(Superknn(epsilon=self.epsilon, models=self.models,
                                                  regression=self.regression, metric=self.metric,
                                                shuffle= self.shuffle, needs_proba=self.needs_proba,
                                                n_jobs=1, verbose=False,
                                                  folds=self.folds, random_state=self.random_state),
                                             return_train_score=False,
                                             param_distributions=tuned_parameters, n_iter=n_iter_search,

                                             scoring="accuracy")
                else:
                    clf = GridSearchCV(Superknn(epsilon=self.epsilon, models=self.models,
                                                  regression=self.regression, metric=self.metric,
                                                shuffle=self.shuffle, folds=self.folds,
                                                needs_proba=self.needs_proba,
                                                n_jobs=1, verbose=False,
                                                random_state=self.random_state),
                                             return_train_score=False,
                                             param_grid=tuned_parameters,
                                             scoring="accuracy")
            clf.fit(X_opt, y_opt)
            if self.regression or self.needs_proba:
                self.epsilon = clf.best_params_["epsilon"]
                print("optimal epsilon = ", self.epsilon, "\n")
            self.alpha = clf.best_params_["alpha"]
            print("optimal alpha = ", self.alpha, "\n")

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: array-like, [n_samples, n_features]
            Training data which will be used to create the COBRA aggregate.

        y: array-like, shape = [n_samples]
            Target values used to train the machines used in the aggregation.

        X_k : shape = [n_samples, n_features]
            Training data which is used to train the machines used in the aggregation.
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_k : array-like, shape = [n_samples]
            Target values used to train the machines used in the aggregation.

        X_l : shape = [n_samples, n_features]
            Training data which is used to form the aggregate.
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_l : array-like, shape = [n_samples]
            Target values which are actually used to form the aggregate.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        # self.machines_array = []
        # for model in self.models:
        #     self.machines_ = {}
        #     self.machines_[model.__class__.__name__] = model.fit(self.X_, self.y_)
        #     self.machines_array.append(self.machines_)
        # return self

    def pred(self, dict):

        global result
        ind = list(dict.keys())[0]
        pred_X = np.array(list(dict.values())[0]).reshape(1, -1)
        # print("pred_X:", pred_X)

        points = []
        # count is the indice number.
        for count in range(0, len(self.stack_X)):
            if self.regression:
                distance = find_distance(self.stack_X[count], pred_X.ravel(), self.epsilon)
            else:
                if self.needs_proba:
                    distance = find_distance(self.stack_X[count], pred_X.ravel(), self.epsilon)
                else:
                    distance = find_equal(self.stack_X[count], pred_X.ravel())

            if distance >= self.alpha:
                points.append(count)

        # if no points are selected, return 0
        if len(points) == 0:
            if self.verbose:
                print(ind, "No points were selected, prediction is 0")
            result.put({ind:np.array([0])})
        else:
            '''定义距离，对于回归直接用距离与weight成反比，距离用空间距离'''
            if self.regression:
                neigh = KNeighborsRegressor(n_neighbors=len(points), weights='distance')
                neigh.fit(self.stack_X[points], self.y_[points])
                avg = neigh.predict(pred_X)
                op = {ind: avg}
                result.put(op)
            else:
                neigh = KNeighborsClassifier(n_neighbors=len(points), weights='distance')
                neigh.fit(self.stack_X[points], self.y_[points])
                avg = neigh.predict(pred_X)
                op = {ind: avg}
                result.put(op)


    def predict(self, X_test):

        # sets alpha as the total number of machines as a default value
        if self.alpha is None:
            self.alpha = len(self.models)
        print("alpha:", self.alpha)
        if self.epsilon:
            print("epsilon:", self.epsilon)
        X_test = check_array(X_test)

        start_time = time.time()
        stack_model = Stacking(X_train=self.X_, y_train=self.y_, X_test=X_test, bagged_pred=False,
                               regression=self.regression, metric=self.metric,
                               n_folds=self.folds, needs_proba=self.needs_proba,
                               shuffle=self.shuffle, random_state=self.random_state, verbose=0)
        stack_model.add(self.models)
        print('stacking took %fs!' % (time.time() - start_time))
        self.stack_X = stack_model.next_train
        X_test_new = stack_model.next_test


        start_time = time.time()
        if X_test_new.ndim == 1:
            return self.pred(X_test.reshape(1, -1))

        b = []
        p = Pool(self.n_jobs)
        dict_array = []
        for index, vector in enumerate(X_test_new):
            dict = {}
            dict[index] = vector
            dict_array.append(dict)
        p.map(self.pred, dict_array)
        p.close()
        while (result.qsize() > 0):
            b.append(result.get())
        dic = {}
        pred = []
        for data in b:
            key = list(data.keys())[0]
            value = list(data.values())[0]
            dic[key] = value
        for key in sorted(dic.keys()):
            pred.append(dic[key])
        print('predict took %fs!\n' % (time.time() - start_time))
        return pred





