import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from autotf.feature_engineering.feature_selection.wrapper import sffs

d = 5

dataset = np.load("../../data/madelon.npz")
x, y = dataset['x'], dataset['y']

clf = KNeighborsClassifier(n_neighbors=3)

selected = sffs(x, y, d, clf)
