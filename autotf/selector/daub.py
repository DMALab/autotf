import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class DAUB:
    """
    DAUB(Data Allocation using Upper Bounds) is a method for searching the best algorithm within
    limited time.
    Reference:
    Sabharwal A, Samulowitz H, Tesauro G (2016) Selecting near-optimal learners via incremental data allocation.
    Proc. AAAI 2016.
    """

    def __init__(self, learners, x, y, task_type='classification', ratio=1.5, b=200):
        self.learners = learners
        self.x = x
        self.y = y
        self.task_type = task_type
        self._ratio = ratio
        self._b = b

        self.score_validation = []
        self.score_train = []
        for k in range(len(learners)):
            self.score_validation.append([])
            self.score_train.append([])

        self._N = None

    def fit(self):
        M = len(self.learners)
        self._N = int(0.8 * len(self.x))
        n_samples = []
        upper_bounds = []

        # initialization with 3 points
        for i in range(M):
            for k in range(3):
                self.train_learner(i, int(self._b * self._ratio ** k))
            n_samples.append(self._ratio ** 2 * self._b)
            upper_bounds.append(self.update_bound(i, n_samples[i]))

        # iteration
        while max(n_samples) < self._N:
            j = np.argmax(upper_bounds)
            print('j = ', j)
            n_samples[j] = min(int(self._ratio * n_samples[j]), self._N)
            print('n_samples = ', n_samples[j])
            self.train_learner(j, n_samples[j])
            upper_bounds[j] = self.update_bound(j, n_samples[j])
        for i in range(len(self.learners)):
            if n_samples[i] == self._N:
                return i, self.score_validation[i][-1]

    def train_learner(self, i, n_samples):

        k_fold = StratifiedKFold(n_splits=5)
        accu = []
        accu_train = []
        for train_index, test_index in k_fold.split(self.x, self.y):
            bootstrap = np.random.randint(0, len(train_index), size=(n_samples,))
            model = self.learners[i]
            model.fit(self.x[train_index][bootstrap], self.y[train_index][bootstrap])

            y_pred = model.predict(self.x[test_index])
            y_pred_train = model.predict(self.x[train_index][bootstrap])

            accu.append(accuracy_score(self.y[test_index], y_pred))
            accu_train.append(accuracy_score(self.y[train_index][bootstrap], y_pred_train))

        accu_mean = np.average(accu)
        if len(self.score_validation[i]) >= 1:
            detla = accu_mean - self.score_validation[i][-1]
            if detla < 0:
                self.score_validation[i][-1] += (detla / 2)
                accu_mean -= (detla / 2)
        accu_mean_train = np.average(accu_train)

        self.score_validation[i].append(accu_mean)
        self.score_train[i].append(accu_mean_train)

    def update_bound(self, i, n_samples):
        A = np.array([
            [n_samples, 1.0],
            [n_samples / self._ratio, 1.0],
            [n_samples / self._ratio ** 2, 1.0]
        ])
        y = np.array([self.score_validation[i][-1],
                      self.score_validation[i][-2],
                      self.score_validation[i][-3]
                      ])
        slope, _ = np.linalg.lstsq(A, y, None)[0]
        # print('slope=', slope)
        upper = slope * (self._N - n_samples) + self.score_validation[i][-1]

        return min(self.score_train[i][-1], upper)


