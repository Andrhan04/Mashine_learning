import numpy as np


def gaussian(x, a, q):
    q_eps = q + 1e-9
    return (np.exp(-((x - a) ** 2) / (2 * q_eps))) / (np.sqrt(2 * np.pi * q_eps))

def laplace(x, a, b):
    b_eps = b + 1e-9
    return (np.exp(-np.abs(x - a) / b_eps)) / (2 * b_eps)