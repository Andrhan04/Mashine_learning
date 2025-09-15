from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def load_dataset():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    uniq_y = np.unique(y)
    num_y = np.array([np.where(uniq_y == val)[0][0] for val in y])
    X_train, X_test, y_train, y_test = train_test_split(X, num_y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test