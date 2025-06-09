import numpy as np
from Class.generalizing_ability import Cross_validation
def predict(X, weights):
    return X.dot(weights)

eps=1e-5

def main(X, Y, lambda_reg : float, exp=0.1, iter = 1000):
    m, n = X.shape  
    weights = np.random.uniform(-1/(2*n), 1/(2*n), n)  
    Q = Cross_validation(X, Y, weights, lambda_reg)
    exp = exp
    cnt = 0
    mem_weights = weights

    for step in range(iter):

        i = np.random.randint(m)
    
        X_i = X[i]
        y_i = Y[i]
    
        prediction = X_i.dot(weights)
    
        error = prediction - y_i
        gradient = 2*X_i.T.dot(error) + 2 * lambda_reg * weights
    
        weights -= (1 / (step + 1)) * gradient

        curr_Q = error * exp - (1 - exp) * Q

        if np.linalg.norm(weights - mem_weights) < eps:
            cnt += 1
            if cnt == 100:
                mem_weights = weights
                break
        else: 
            cnt == 0
    return mem_weights