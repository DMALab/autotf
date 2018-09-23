from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ensemble import Stacking
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss,accuracy_score

n_classes = 3

# Create data: 500 example, 5 feature, 3 classes
X, y = make_classification(n_samples=500, n_features=5,
                           n_informative=3, n_redundant=1,
                           n_classes=n_classes, flip_y=0,
                           random_state=0)

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Train shape:', X_train.shape)
print('Test shape: ', X_test.shape)

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

ensemble = Stacking(X_train, y_train, X_test, regression=False, bagged_pred=True,
                    needs_proba=True, save_dir=None, metric=log_loss,
                    n_folds=4, stratified=True, shuffle=True,
                    random_state=0, verbose=2)

ensemble.add(models, propagate_features=[0, 1,2,3,4])

y_pred = ensemble.add_meta(meta_model)

print('Final prediction score: %.8f' % accuracy_score(y_test, y_pred))


'''Compare with the randomforest'''

print("randomforest")
model = RandomForestClassifier(random_state=0, n_jobs=-1,
                           n_estimators=500, max_depth=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))