from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import GridSearchCV
import math
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger('ensemble.cobra')


class Cobra(BaseEstimator):
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

    def __init__(self,X = None, y=None, regression=True, metric = None, shuffle=None, random_state=None, epsilon=None,
                 alpha=None, models=None, k=None):
        self.random_state = random_state
        self.epsilon = epsilon
        self.alpha = alpha
        self.models = models
        self.shuffle = shuffle
        self.metric = metric
        self.k = k
        self.X_ = X
        self.y_ = y
        self.regression = regression


    def opmimal_parameters(self, X = None, y=None, eps_size=None, grid_points=None, verbose = 0):
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

        _, X_eps, _, y_eps = train_test_split(X, y, test_size=eps_size, shuffle=self.shuffle, random_state=self.random_state)
        print("Number of optimal data:", len(y_eps), "\n")
        if self.regression:
            self.set_epsilon(X_epsilon=X_eps, y_epsilon=y_eps, grid_points=grid_points,
                             verbose=verbose)
        self.set_alpha(X_epsilon=X_eps, y_epsilon=y_eps,verbose=verbose)
        self.set_split(X_epsilon=X_eps, y_epsilon=y_eps)

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
        self.machines_ = {}


        if self.k is None:
            self.split_data()
            self.load_machines()
            self.load_machine_predictions()
        else:
            k = int(self.k * len(self.X_))
            l = len(self.X_)
            self.X_k_ = self.X_[:k]
            self.X_l_ = self.X_[k:l]
            self.y_k_ = self.y_[:k]
            self.y_l_ = self.y_[k:l]
            self.load_machines()
            self.load_machine_predictions()
        return self


    def set_epsilon(self, X_epsilon=None, y_epsilon=None, grid_points=None, verbose=0):
        """
        Parameters
        ----------

        X_epsilon : shape = [n_samples, n_features]
            Used if no epsilon is passed to find the optimal epsilon for data passed.

        y_epsilon : array-like, shape = [n_samples]
            Used if no epsilon is passed to find the optimal epsilon for data passed.

        grid_points: int, optional
            If no epsilon value is passed, this parameter controls how many points on the grid to traverse.
   
        """
        # set up COBRA to perform CV and find an optimal epsilon.
        print("---------Finding the optimal epsilon---------")
        if self.epsilon is None and X_epsilon is not None:
            self.X_ = X_epsilon
            self.y_ = y_epsilon

            self.split_data()
            self.load_machines()
            self.load_machine_predictions()

            # get the candidate epsilons
            a, size = sorted(self.all_predictions_), len(self.all_predictions_)
            res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
            emin = min(res)
            emax = max(a) - min(a)
            erange = np.linspace(emin, emax, grid_points)
            tuned_parameters = [{'epsilon': erange}]
            print(tuned_parameters)
            clf = GridSearchCV(Cobra(epsilon=None, models=self.models, regression=True,
                                         random_state=self.random_state), tuned_parameters,
                               return_train_score=False, verbose=verbose,
                                   cv=5, scoring="neg_mean_squared_error")
            clf.fit(X_epsilon, y_epsilon)
            self.epsilon = clf.best_params_["epsilon"]
            self.machines_, self.machine_predictions_ = {}, {}
            print("optimal epsilon = ", self.epsilon, "\n")

    def set_alpha(self, X_epsilon=None, y_epsilon=None,verbose=0):
        print("---------Finding the optimal alpha---------")
        if self.alpha is None and X_epsilon is not None:
            self.X_ = X_epsilon
            self.y_ = y_epsilon
            arange = range(1, len(self.models) + 1)
            tuned_parameters = [{'alpha': arange}]
            if self.regression:
                clf = GridSearchCV(Cobra(epsilon=self.epsilon, models=self.models, regression=True,
                                         random_state=self.random_state),
                                    tuned_parameters,return_train_score=False,verbose=verbose,
                                   cv=5, scoring="neg_mean_squared_error")
            else:
                clf = GridSearchCV(Cobra(epsilon=self.epsilon, models=self.models, regression=False,
                                         random_state=self.random_state),
                                   tuned_parameters, cv=5, scoring="accuracy")
            clf.fit(X_epsilon, y_epsilon)
            self.alpha = clf.best_params_["alpha"]
            self.machines_, self.machine_predictions_ = {}, {}
            print("omtimal alpha = ", self.alpha, "\n")

    def set_split(self, X_epsilon=None, y_epsilon=None):
        print("---------Finding the optimal k---------")
        X_eps_train, X_eps_pre, y_eps_train, y_eps_pre = train_test_split(X_epsilon, y_epsilon, test_size=0.2, shuffle=self.shuffle,
                                              random_state=self.random_state)
        split = [(0.20, 0.80), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.80, 0.20)]
        Score= {}
        for k, l in split:
            # print('k = ',k)
            if self.regression:
                machine = Cobra(X=X_eps_train, y=y_eps_train, epsilon=self.epsilon, models=self.models,
                                random_state=self.random_state,regression=True,
                                alpha=self.alpha)
            else:
                machine = Cobra(X=X_eps_train, y=y_eps_train, epsilon=self.epsilon, models=self.models,
                                random_state=self.random_state,regression=False,
                                alpha=self.alpha)

            machine.split_data(int(k * len(X_eps_train)), int((k + l) * len(X_eps_train)))

            machine.load_machines()

            machine.load_machine_predictions()

            results = machine.predict(X_eps_pre)
            Score[(k, l)] = (self.metric(y_eps_pre, results))
        if self.regression:
            opt = min(Score, key=Score.get)
        else:
            opt = max(Score, key=Score.get)
        self.k = opt[0]
        print("optimal k = ", self.k, "\n")

    def pred(self, X, alpha, info=False):
        """
        Performs the COBRA aggregation scheme, used in predict method.

        Parameters
        ----------
        X: array-like, [n_features]

        alpha: int, optional
            alpha refers to the number of machines the prediction must be close to to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        avg: prediction

        """
        # dictionary mapping machine to points selected
        select = {}
        for machine in self.machines_:
            # machine prediction
            val = self.machines_[machine].predict(X)
            select[machine] = set()
            # iterating from l to n
            # replace with numpy iteration
            for count in range(0, len(self.X_l_)):
                if self.regression:
                    try:
                        # if value is close to prediction, select the indice
                        if math.fabs(self.machine_predictions_[machine][count] - val) <= self.epsilon:
                            select[machine].add(count)
                    except (ValueError, TypeError) as e:
                        logger.info("Error in indice selection")
                        continue
                else:
                    if self.machine_predictions_[machine][count] == val:
                        select[machine].add(count)

        points = []
        # count is the indice number.
        for count in range(0, len(self.X_l_)):
            # row check is number of machines which picked up a particular point
            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check >= alpha:
                points.append(count)

        # if no points are selected, return 0
        if len(points) == 0:
            if info:
                logger.info("No points were selected, prediction is 0")
                return (0, 0)
            return 0

        # aggregate
        if self.regression:
            avg = 0
            for point in points:
                avg += self.y_l_[point]
            avg = avg / len(points)
            if info:
                return avg, points
            return avg
        else:
            classes = {}
            for label in np.unique(self.y_l_):
                classes[label] = 0

            for point in points:
                classes[self.y_l_[point]] += 1
            result = int(max(classes, key=classes.get))
            if info:
                return result, points
            return result




    def predict(self, X, info=False):
        """
        Performs the COBRA aggregation scheme, calls pred.

        Parameters
        ----------
        X: array-like, [n_features]

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        result: prediction

        """

        # sets alpha as the total number of machines as a default value

        X = check_array(X)
        # sets alpha as the total number of machines as a default value
        if self.alpha is None:
            self.alpha = len(self.models)
        # print("alpha:", self.alpha)
        # if self.epsilon is not None:
        #     print("epsilon:", self.epsilon)
        if X.ndim == 1:
            return self.pred(X.reshape(1, -1), info=info, alpha=self.alpha)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pred(vector.reshape(1, -1), info=info, alpha=self.alpha)
                avg_points += len(points)
            else:
                result[index] = self.pred(vector.reshape(1, -1), info=info, alpha=self.alpha)
            index += 1

        if info:
            avg_points = avg_points / len(X)
            return result, avg_points

        return result


    def split_data(self, k=None, l=None):
        """
        Split the data into different parts for training machines and for aggregation.

        Parameters
        ----------
        k : int, optional
            k is the number of points used to train the machines.
            Those are the first k points of the data provided.

        l: int, optional
            l is the number of points used to form the COBRA aggregate.

        shuffle: bool, optional
            Boolean value to decide to shuffle the data before splitting.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.shuffle:
            self.X_, self.y_ = shuffle(self.X_, self.y_, random_state=self.random_state)

        if k is None and l is None:
            k = int(len(self.X_) / 2)
            l = int(len(self.X_))

        if k is not None and l is None:
            l = len(self.X_) - k

        if l is not None and k is None:
            k = len(self.X_) - l

        self.X_k_ = self.X_[:k]
        self.X_l_ = self.X_[k:l]
        self.y_k_ = self.y_[:k]
        self.y_l_ = self.y_[k:l]

        return self


    def load_machines(self):
        """Fit the machines"""
        self.machines_ = {}
        for model in self.models:
            self.machines_[model.__class__.__name__] = model.fit(self.X_k_, self.y_k_)
        return self


    def load_machine_predictions(self):
        """
        Stores the trained machines' predicitons on training data in a dictionary, to be used for predictions.
        Should be run after all the machines to be used for aggregation is loaded.

        Parameters
        ----------
        predictions: dictionary, optional
            A pre-existing machine:predictions dictionary can also be loaded.

        Returns
        -------
        self : returns an instance of self.
        """
        self.machine_predictions_ = {}
        self.all_predictions_ = np.array([])
        for machine in self.machines_:
            self.machine_predictions_[machine] = self.machines_[machine].predict(self.X_l_)
            self.all_predictions_ = np.append(self.all_predictions_, self.machine_predictions_[machine])
        return self
