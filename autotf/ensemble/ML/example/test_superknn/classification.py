from ensemble.superknn import Superknn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import time
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# Prepare data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)
'''实验1：不对model调参'''
'''1.just use the XGB'''
model = XGBClassifier(n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use XGB): [%.8f]' % accuracy_score(y_test, y_pred))

'''2.just use the Extra'''
model = ExtraTreesClassifier(n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use Extra): [%.8f]' % accuracy_score(y_test, y_pred))

'''3.just use the GBDT'''
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use GBDT): [%.8f]' % accuracy_score(y_test, y_pred))

'''4.just use the RF'''
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use RF): [%.8f]' % accuracy_score(y_test, y_pred))

'''use KNN'''
model = KNeighborsClassifier(n_jobs=-1)
start_time = time.time()
model.fit(X_train, y_train)
print ('training(KNN) took %fs!' % (time.time() - start_time))
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use KNN): [%.8f]' % accuracy_score(y_test, y_pred))

'''5.use superknn'''
# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

models = [ExtraTreesClassifier(n_jobs=-1),
    RandomForestClassifier(n_jobs=-1),
    GradientBoostingClassifier(),
    XGBClassifier(n_jobs=-1)
]


'''方法1：不适用need_proba,这时只有参数alpha,所以不需要调用opmimal_parameters函数'''
# model = Superknn(models=models, metric=accuracy_score, regression=False, random_state=0,
#                  folds=5, needs_proba=False, shuffle=True)
# # #一般来说，此时参数比较少可以一个个试alpha，不需要调用optimal_parametersh函数
# # model.opmimal_parameters(X=X_train, y=y_train, eps_size=0.05)
# alphas = np.arange(2, 5, 1)
# for alpha in alphas:
#     print('alpha:', alpha)
#     model.alpha = alpha
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     # print(y_test)
#     # print(y_pred)
#     print('Final prediction score with optimal(use superknn): [%.8f]' % accuracy_score(y_test, y_pred))

'''方法2：使用need_proba,这时只有参数alpha和epsilon，其中alpha一共有models*classes = 4*10=40个,
grid_points = 10个
所以先调用opmimal_parameters函数先找到一个大概的范围'''
model = Superknn(models=models, metric=accuracy_score, regression=False, random_state=0,
                 folds=5, needs_proba=True, shuffle=True)
# model.opmimal_parameters(X=X_train, y=y_train, eps_size=0.05, grid_points=10)
epsilons = np.arange(0.08, 0.15, 0.01)
model.alpha = 26
for epsilon in epsilons:
    print('epsilon:', epsilon)
    model.epsilon = epsilon

    start_time = time.time()
    model.fit(X_train, y_train)
    print('fit took %fs!' % (time.time() - start_time))

    start_time = time.time()
    y_pred = model.predict(X_test)
    print('predict took %fs!' % (time.time() - start_time))
    # print(y_test)
    # print(y_pred)
    print('Final prediction score with optimal(use superknn): [%.8f]' % accuracy_score(y_test, y_pred))

