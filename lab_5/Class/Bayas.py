import numpy as np

def fit_gaus(X, y):
    unic_y = np.unique(y)
    M = {}
    D = {}
    aprior = {}
    for i in unic_y:
        X_c = X[y == i]
        M[i] = np.mean(X_c, axis=0)
        D[i] = np.var(X_c, axis=0)
        aprior[i] = len(X_c) / len(y)
    return M, D, aprior, unic_y

def fit_laplace(X, y):
    unic_y = np.unique(y)
    M = {}
    B = {}
    aprior = {}
    for i in unic_y:
        X_c = X[y == i]
        M[i] = np.mean(X_c, axis=0)
        B[i] = np.mean(np.abs(X_c - M[i]), axis=0)
        aprior[i] = len(X_c) / len(y)
    return M, B, aprior, unic_y

def pred(X, M, D, aprior, unic_y, func):
    pred = []
    for x in X:
        aposter = []
        for y_class in unic_y:
            prior = np.log(aprior[y_class])
            log_true = np.sum(np.log(func(x, M[y_class], D[y_class])))
            poster = prior + log_true
            aposter.append(poster)
        pred.append(unic_y[np.argmax(aposter)])
    return np.array(pred)
