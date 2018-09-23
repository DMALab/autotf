import time
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ensemble.superknn import Superknn
from ensemble.stack import Stacking
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
data = pd.read_csv('transfusion.csv')
X_index = range(data.values.shape[1]-1)
X = data.values[:, X_index]
Y = data.values[:, -1]
train_x,test_x, train_y, test_y = train_test_split(X,  Y, stratify=Y, shuffle=True,
                                                   test_size = 0.2,
                                                   random_state = 0)
num_train, num_feat = train_x.shape
num_test, num_feat = test_x.shape
is_binary_class = (len(np.unique(train_y)) == 2)
print ('******************** Data Info *********************' )
print( '#training data: %d, #testing_data: %d, dimension: %d\n' % (num_train, num_test, num_feat))

# print ('******************** 1.KNN *********************' )
# start_time = time.time()
# model = KNeighborsClassifier(n_jobs=-1, weights='distance')
# model.fit(train_x, train_y)
# print ('training(KNN) took %fs!' % (time.time() - start_time))
# start_time = time.time()
# pred_y = model.predict(test_x)
# print ('predict(KNN) took %fs!' % (time.time() - start_time))
# accuracy = metrics.accuracy_score(test_y, pred_y)
# print ('accuracy(KNN): %.2f%%\n' % (100 * accuracy))
#
# print ('******************** 2.Cart *********************' )
# start_time = time.time()
# model = DecisionTreeClassifier()
# model.fit(train_x, train_y)
# print ('training(Cart) took %fs!' % (time.time() - start_time))
# start_time = time.time()
# pred_y = model.predict(test_x)
# print ('predict(Cart) took %fs!' % (time.time() - start_time))
# accuracy = metrics.accuracy_score(test_y, pred_y)
# print ('accuracy(Cart): %.2f%%\n' % (100 * accuracy))
#
# print ('******************** 3.SVM  *********************' )
# start_time = time.time()
# model = svm.SVC(random_state=0, probability=True)
# model.fit(train_x, train_y)
# print ('training(SVM) took %fs!' % (time.time() - start_time))
# start_time = time.time()
# pred_y = model.predict(test_x)
# a = model.predict_proba(test_x)
# print ('predict(SVM) took %fs!' % (time.time() - start_time))
# accuracy = metrics.accuracy_score(test_y, pred_y)
# print ('accuracy(SVM): %.2f%%\n' % (100 * accuracy))
#
# print ('******************** 4.NB *********************' )
# start_time = time.time()
# model = MultinomialNB()
# model.fit(train_x, train_y)
# print ('training(NB) took %fs!' % (time.time() - start_time))
# start_time = time.time()
# pred_y = model.predict(test_x)
# print ('predict(NB) took %fs!' % (time.time() - start_time))
# accuracy = metrics.accuracy_score(test_y, pred_y)
# print ('accuracy(NB): %.2f%%\n' % (100 * accuracy))
#
# print ('******************** 5.Stacking *********************' )
# models = [KNeighborsClassifier(n_jobs=-1, weights='distance'),
#           DecisionTreeClassifier(random_state=0),
#           svm.SVC(random_state=0, probability=True),
#           MultinomialNB()
# ]
# meta_model = LogisticRegression(random_state=0)
# start_time = time.time()
# ensemble = Stacking(train_x, train_y, test_x, regression=False, bagged_pred=True,
#                     needs_proba=True, save_dir=None, metric=accuracy_score,
#                     n_folds=4, stratified=True, shuffle=True,
#                     random_state=0, verbose=0)
# start_time = time.time()
# ensemble.add(models)
# y_pred = ensemble.add_meta(meta_model)
# print ('process(add) took %fs!' % (time.time() - start_time))
# print('Final prediction score with optimal(use superknn): [%.8f]\n' % accuracy_score(test_y, y_pred))
#

print ('******************** 6.Superknn *********************' )
models = [KNeighborsClassifier(n_jobs=1, weights='distance'),
          DecisionTreeClassifier(random_state=0),
          svm.SVC(random_state=0, probability=True),
          MultinomialNB()
]
model = Superknn(models=models, metric=accuracy_score, regression=False, random_state=0,
                 folds=5, needs_proba=True, shuffle=True)
model.opmimal_parameters(X=train_x, y=train_y, eps_size=0.2, grid_points=10)
# '''Getting parameters'''
# arange = range(1, len(models)*len(set(train_y)) + 1)
#
# stack_model = Stacking(X_train=train_x, y_train=train_y, X_test=None, bagged_pred=False,
#                                            regression=model.regression, needs_proba=model.needs_proba,
#                                            metric=model.metric, n_folds=model.folds,
#                                            shuffle=True, random_state=0, verbose=0)
# stack_model.add(models)
# stack_X = stack_model.next_train
# a, size = sorted(stack_X.ravel()), len(stack_X.ravel())
# res = [a[i + 1] - a[i] for i in range(size) if i + 1 < size]
# emin = min(res)
# emax = max(a) - min(a)
# erange = np.linspace(emin, emax, 5)
# print('erange:', erange)
# print('arange:', arange)
# #1-调参, gridsearch
# param_test1 = {
#  'alpha': [2, 3, 4],
#  'epsilon': erange
# }
# clf = GridSearchCV(model, return_train_score=False,
#                     param_grid = param_test1, iid=False, cv=5,
#                          scoring="accuracy")
# clf.fit(train_x, train_y)
# print(clf.grid_scores_)
# print('---------')
# print(clf.best_params_)
# print('---------')
# print(clf.best_score_)



#
# model.alpha =5
# model.epsilon=0.0
# model.fit(train_x, train_y)
# y_pred = model.predict(test_x)
# print('Final prediction score with optimal(use superknn): [%.8f]\n' % accuracy_score(test_y, y_pred))


epsilons = np.arange(0.1, 1, 0.1)
model.alpha = 4
for epsilon in epsilons:
    model.epsilon = epsilon
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    print('Final prediction score with optimal(use superknn): [%.8f]\n' % accuracy_score(test_y, y_pred))
