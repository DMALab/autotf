import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score
from ensemble import Stacking
from sklearn.linear_model import LogisticRegression

class DeepSuperLearner(BaseEstimator):
    '''
    DeepSuperLearner ensemble method of learners for classification.
    Parameters
    ----------
    blearner: python dictionary of learner name with its instance. {'SVM':svm_instance} for instance.
    Attributes
    ----------
    K: KFolds integer used for training.
    '''

    def __init__(self, models, k=5, max_iter=None, shuffle=False, weight_on_model=False, random_state=0):
        self.models = models
        self.folds = k
        self.random_state = random_state
        self.weight_on_model = weight_on_model
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.coef_optimization_method = 'SLSQP'
        self.n_baselearners = len(models)
        self.trim_eps = 1e-5
        self.trim_func = lambda x: np.clip(x, self.trim_eps, 1 - self.trim_eps)
        self.weights_per_iteration = []
        self.fitted_learners_per_iteration = []
        self.__classes_n = 0
        self.label_onehotencoder = LabelBinarizer()

    def _get_weighted_prediction(self, m_set_predictions, weights):
        """
        Calculate weighted combination of predictions probabilities
        Parameters
        ----------
        m_set_predictions: numpy.array of shape [n, m, j]
                    where each column is a vector of j-class probablities 
                    from each base learner (each channel represent probability of
                    different class).
        weights: numpy.array of length m (base learners count),
        to be used to combine columns of m_set_predictions.
        Returns
        _______
        avgprobs: numpy.array of shape [n,j].


        """
        trimp = self.trim_func(m_set_predictions)
        weights_probs = np.stack([np.dot(trimp[:, :, i], weights)
                                  for i in range(trimp.shape[-1])]).T
        return weights_probs

    def _get_logloss(self, y, y_pred, sample_weight=None):
        """
        Calculate the normalized logloss given ground-truth y and y-predictions
        Parameters
        ----------
        y: numpy array of shape [n,j] (ground-truth)
        y_pred: numpy array of shape [n,j] (predictions)

        Attributes
        ----------
        sample_weight: numpy array of shape [n,]

        Returns
        -------
        Logloss: estimated logloss of ground-truth and predictions.
        """
        return log_loss(y, y_pred, eps=self.trim_eps,
                        sample_weight=sample_weight)

    def _get_weights(self, y, m_set_predictions_fold):
        """
        Find weights that minimize the estimated logloss.
        Parameters
        ----------
        y: numpy.array of shape [n,j]
        m_set_predictions_fold: numpy.array of shape [n, m, j] of fold-k
        Returns
        _______
        weights: numpy.array of normalized non-negative weights to combine
              base learners
        """
        def objective_f(w):  # Logloss(y,w*y_pred)
            return self._get_logloss(y, self._get_weighted_prediction(m_set_predictions_fold, w))

        def normalized_constraint(w):  # Sum(w)-1 == 0
            return np.array([np.sum(w) - 1])

        w0 = np.array([1. / self.n_baselearners] * self.n_baselearners)
        wbounds = [(0, 1)] * self.n_baselearners
        out, _, _, imode, _ = fmin_slsqp(objective_f, \
                                         w0, f_eqcons=normalized_constraint, bounds=wbounds, \
                                         disp=0, full_output=1)
        if imode is not 0:
            raise Exception("Optimization failed to find weights")

        out = np.array(out)
        out[out < np.sqrt(np.finfo(np.double).eps)] = 0
        weights = out / np.sum(out)
        return weights

    def fit(self, X, y):
        n, j = len(y), len(np.unique(y))
        self.__classes_n = j
        X, y = check_X_y(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X, need_proba = False):
        pre_test = []
        self.X_test = check_array(X)
        latest_loss = np.finfo(np.double).max
        for iteration in range(self.max_iter):
            print('X_train.shape:', self.X_train.shape)
            print('X_test.shape:', self.X_test.shape)
            stack_model = Stacking(X_train=self.X_train, y_train=self.y_train, X_test=self.X_test,
                                   bagged_pred=True, regression=False,
                                   n_folds=self.folds, needs_proba=True,
                                   shuffle=self.shuffle, random_state=self.random_state, verbose=0)
            stack_model.add(self.models)
            if self.weight_on_model:
                stack_model.next_train = stack_model.next_train.reshape(len(stack_model.next_train),
                                                    self.n_baselearners,self.__classes_n)
                stack_model.next_test = stack_model.next_test.reshape(len(stack_model.next_test),
                                                    self.n_baselearners, self.__classes_n)
                tmp_weights = self._get_weights(self.y_train, stack_model.next_train)
                avg_probs_train = self._get_weighted_prediction(stack_model.next_train, tmp_weights)
                avg_probs_test = self._get_weighted_prediction(stack_model.next_test, tmp_weights)
                loss = self._get_logloss(self.y_train, avg_probs_train)
                acc = accuracy_score(self.y_train, np.argmax(avg_probs_train, axis=1))
                print("Iteration: {}, Loss: {}, Acc: {}".format(iteration, loss, acc))
                print("Weights: ", tmp_weights, '\n')
                if loss < latest_loss:
                    latest_loss = loss
                    pre_test.append(avg_probs_test)
                    self.X_train = np.hstack((self.X_train, avg_probs_train))
                    self.X_test = np.hstack((self.X_test, avg_probs_test))

                else:
                    if need_proba:
                        return avg_probs_test
                    else:
                        avg_test = np.argmax(pre_test[-1], axis=1)
                        return avg_test
            else:
                meta_model = LogisticRegression(random_state=0)
                meta_model.fit(stack_model.next_train, self.y_train)
                avg_probs_train = meta_model.predict_proba(stack_model.next_train)
                avg_probs_test = meta_model.predict_proba(stack_model.next_test)
                avg_test = meta_model.predict(stack_model.next_test)
                loss = self._get_logloss(self.y_train, avg_probs_train)
                acc = accuracy_score(self.y_train, np.argmax(avg_probs_train, axis=1))
                print("Iteration: {}, Loss: {}, Acc: {}".format(iteration, loss, acc))
                if loss < latest_loss:
                    latest_loss = loss
                    pre_test.append(avg_probs_test)
                    pre_test.append(avg_test)
                    self.X_train = np.hstack((self.X_train, avg_probs_train))
                    self.X_test = np.hstack((self.X_test, avg_probs_test))

                else:
                    if need_proba:
                        return avg_probs_test
                    else:
                        avg_test = pre_test[-1]
                        return avg_test
