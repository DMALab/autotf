from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _evaluate(x, y, clf, loss, selected):
    if loss == "holdout":
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, stratify=y)
        clf.fit(x_train[:, selected], y_train)
        y_pred = clf.predict(x_valid[:, selected])
        return accuracy_score(y_valid, y_pred)
    if loss == "cross_validation":
        pass


def sfs(x, y, k, clf):
    feature_num = x.shape[1]
    selected = []
    remain = list(range(feature_num))

    while len(selected) < k:
        best_perf = -1
        best_feature = None
        for f in remain:
            selected_tmp = list(selected)
            selected_tmp.append(f)
            perf = _evaluate(x, y, clf, "holdout", selected_tmp)
            if perf > best_perf:
                best_perf = perf
                best_feature = f
        selected.append(best_feature)
        remain.remove(best_feature)

        if len(selected) % 5 == 0:
            print("features =", len(selected), "validation_perf =", best_perf)

    return selected


def find_worst_feature(data, labels, clf, selected):
    best_perf = -1
    worst_feature = None
    for f in selected:
        selected_tmp = list(selected)
        selected_tmp.remove(f)
        if len(selected_tmp) == 0:
            perf = 0
        else:
            perf = _evaluate(data, labels, clf, "holdout", selected_tmp)
        if perf > best_perf:
            worst_feature = f
            best_perf = perf

    return worst_feature, best_perf


def sffs(data, labels, d, clf):
    # initialization
    X = [[]] * (d + 2)
    Jx = [0] * (d + 2)
    Y = list(range(data.shape[1]))
    k = 0

    while k < d:
        # find the most significant feature
        best_perf = -1
        best_feature = None
        remain = [feat for feat in Y if feat not in X[k]]

        for f in remain:
            selected_tmp = list(X[k])
            selected_tmp.append(f)
            perf = _evaluate(data, labels, clf, "holdout", selected_tmp)
            if perf > best_perf:
                best_feature = f
                best_perf = perf

        X[k + 1] = list(X[k])
        X[k + 1].append(best_feature)
        Jx[k + 1] = best_perf
        k = k + 1
        # find the least significant feature
        worst_feature, best_perf = find_worst_feature(data, labels, clf, X[k])
        while best_perf > Jx[k - 1]:
            X[k - 1] = list(X[k])
            X[k - 1].remove(worst_feature)
            Jx[k - 1] = best_perf

            k = k - 1
            worst_feature, best_perf = find_worst_feature(data, labels, clf, X[k])

        if len(X[k]) % 5 == 0:
            print("features =", len(X[k]), "validation_perf =", Jx[k])

    return X[d]
