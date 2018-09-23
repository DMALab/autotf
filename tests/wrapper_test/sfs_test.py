import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from autotf.feature_engineering.feature_selection.wrapper import sfs

d = 5

dataset = np.load("../../data/madelon.npz")
x, y = dataset['x'], dataset['y']

clf = KNeighborsClassifier(n_neighbors=3)

selected = sfs(x, y, d, clf)

print(selected.__class__, selected)
