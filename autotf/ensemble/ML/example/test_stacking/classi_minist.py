import sys
import os
import time
from sklearn import metrics
import numpy as np
import pickle
from ensemble import Stacking
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from ensemble import Stacking
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f, encoding='iso-8859-1')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

data_file = "mnist.pkl.gz"

X_train, y_train, X_test,  y_test = read_data(data_file)

models = [
    ExtraTreesClassifier(random_state=0, n_jobs=-1,
                         n_estimators=100, max_depth=3),

    RandomForestClassifier(random_state=0, n_jobs=-1,
                           n_estimators=100, max_depth=3),

    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]
meta_model = XGBClassifier(random_state=0, n_jobs=5, learning_rate=0.1,
                            n_estimators=100, max_depth=3)
start_time = time.time()

ens = Stacking(X_train, y_train, X_test, regression=False, bagged_pred=True,
               needs_proba=False, save_dir=None, metric=accuracy_score,
               n_folds=4, stratified=True, shuffle=True,
               random_state=0, verbose=0)

start_time = time.time()
ens.add(models)
print('process(add) took %fs!' % (time.time() - start_time))

start_time = time.time()
y_pred = ens.add_meta(meta_model)
print('process(add_meta) took %fs!' % (time.time() - start_time))

print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
