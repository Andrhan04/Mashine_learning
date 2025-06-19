import numpy as np

def predict(X,weights):
        return X.dot(weights)

def Cross_validation(X, y, weights, lambda_reg):
        m = len(y)
        predictions = predict(X,weights)
        error = y - predictions
        return (1 / (2 * m)) * np.sum(error ** 2) + lambda_reg * np.sum(weights ** 2)

# функция оценки качесва - чем ближе к 1, тем лучше
def r2_score(y_true, y_pred):
        # то насколько модель мимо стреляет
        u = np.sum((y_true - y_pred) ** 2)   # сумма квадратов остатков (ошибок модели) — RSS
        # насколько в принципе y разбросаны от среднего
        v = np.sum((y_true - np.mean(y_true)) ** 2)   # сумма квадратов отклонений от среднего — TSS
        return 1 - u / v