from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ensemble import stacking

iris = load_iris()
X, y = iris.data, iris.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [
    ExtraTreesClassifier(random_state=0, n_jobs=-1,
                         n_estimators=100, max_depth=3),

    RandomForestClassifier(random_state=0, n_jobs=-1,
                           n_estimators=100, max_depth=3),

    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]

meta_model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                           n_estimators=100, max_depth=3)

ensemble = stacking(X_train, y_train, X_test, regression=False, bagged_pred=True,
                    needs_proba=False, save_dir=None, metric=accuracy_score,
                    n_folds=4, stratified=True, shuffle=True,
                    random_state=0, verbose=2)

ensemble.add(models, propagate_features=[0, 1])

y_pred = ensemble.add_meta(meta_model)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))