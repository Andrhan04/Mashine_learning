import numpy as np

def predict(X,weights):
        return X.dot(weights)

def Cross_validation(X, y, weights, lambda_reg):
        m = len(y)
        predictions = predict(X,weights)
        error = y - predictions
        return (1 / (2 * m)) * np.sum(error ** 2) + lambda_reg * np.sum(weights ** 2)