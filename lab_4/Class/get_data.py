from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np

def load_and_split_data():
    dataset = load_breast_cancer()
    X = dataset.data
    Y = dataset.target

    return train_test_split(X, Y, stratify = Y, random_state = 42)


def scale_data(X_train, X_test):
    # стандартизация
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def Get_standart_data():
    X_train, X_test, Y_train, Y_test = load_and_split_data()
    X_train, X_test = scale_data(X_train, X_test)
    return X_train, X_test, Y_train, Y_test