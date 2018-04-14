from ensemble import Cobra
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris

# Prepare data
iris = load_iris()
X, y = iris.data, iris.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)
_, X_eps, _, y_eps = train_test_split(X, y, test_size=0.7, random_state=0)


models = [linear_model.LassoCV(), DecisionTreeRegressor(), linear_model.RidgeCV(), RandomForestRegressor()]
'''With optimal'''
cobra = Cobra(models=models, regression=False, metric=accuracy_score, random_state=0, shuffle=True)
cobra.opmimal_parameters(X=X_train, y=y_train, eps_size=0.6, grid_points=15)
cobra.fit(X_train, y_train)
y_pred = cobra.predict(X_test)

print('Final prediction score with optimal: [%.8f]' % accuracy_score(y_test, y_pred))


'''Without optimal'''
cobra = Cobra(models=models,  random_state=0, regression=False, shuffle=True)
cobra.fit(X_train, y_train)
y_pred = cobra.predict(X_test)

print('Final prediction score without optimal: [%.8f]' % accuracy_score(y_test, y_pred))