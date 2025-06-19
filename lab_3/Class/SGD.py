import numpy as np
from Class.generalizing_ability import Cross_validation
def predict(X, weights):
    return X.dot(weights)

eps=1e-5

def main(X, Y, lambda_reg : float, exp : float = 0.1, iter = 10000):
    m, n = X.shape  
    weights = np.random.uniform(-1 / (2*n), 1 / (2*n), n)  
    Q = Cross_validation(X, Y, weights, lambda_reg)
    cnt = 0
    mem_weights = weights
    mem_Q = np.array([])
    mem_Q = np.append(mem_Q,Q)
    for step in range(iter):
        i = np.random.randint(m)
        X_i = X[i]
        y_i = Y[i]
        prediction = X_i.dot(weights)
        error = prediction - y_i
        gradient = 2*X_i.T.dot(error) + 2 * lambda_reg * weights
        weights -= (1 / (step + 1)) * gradient
        curr_Q = error * exp - (1 - exp) * Q
        mem_Q = np.append(mem_Q, curr_Q)
        if (abs(curr_Q - Q)  < eps or np.linalg.norm(weights - mem_weights) < eps):
            cnt += 1
            if cnt == 200:
                mem_weights = weights
                break
        else: 
            cnt = 0
        Q = curr_Q
    return mem_weights, mem_Q