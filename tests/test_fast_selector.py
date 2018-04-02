"""
dataset: openml- MagicTelescope
number of instances: 19020
number of features: 10

duab:
execution time: 36s
cross-validition accuracy(5-fold): 0.87
best model: gbdt

greedy search:
execution time: 150s
cross-validition accuracy(5-fold): 0.87
best model: gbdt

"""

import numpy as np
import pandas as pd

from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from autotf.selector.fast_selector import FastSelector

LEARNER_NAMES = ['logistic', 'svc', 'knn', 'decision_tree', 'adaboost', 'random_forest',
                 'gbdt', 'BernulliNB', 'GaussianNB']

learners = [LogisticRegression(), SVC(), KNeighborsClassifier(),
                DecisionTreeClassifier(), AdaBoostClassifier(), RandomForestClassifier(),
                GradientBoostingClassifier(),
                BernoulliNB(), GaussianNB()]


def greedy_search(x, y):
    t1 = time()
    k_fold = StratifiedKFold(n_splits=5)

    accu_array = []
    for learner in learners:
        print(learner)
        accu = 0
        for train_index, test_index in k_fold.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            learner.fit(x_train, y_train)
            y_pred = learner.predict(x_test)

            accu += accuracy_score(y_test, y_pred)
        accu_array.append(accu / 5.0)

    t2 = time()
    print('execution time:', t2 - t1)

    j = np.argmax(accu_array)
    print('learner = ', LEARNER_NAMES[j], 'accu = ', accu_array[j])


def daub_search(x, y):
    t1 = time()
    selector = FastSelector(task_type='classification')
    learner_num, accu = selector.select_model(x, y)
    t2 = time()
    print('execution time', t2 - t1)
    print('leanrner = ', LEARNER_NAMES[learner_num], 'accu = ', accu)


if __name__ == '__main__':
    df = pd.read_csv('~/datasets/MagicTelescope.csv')
    y = df['class:'].values
    x = df.drop(labels=['ID', 'class:'], axis=1).values

    daub_search(x, y)
    greedy_search(x, y)


