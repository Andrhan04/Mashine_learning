import numpy as np
from Class.generalizing_ability import Cross_validation

def predict(X, weights):
    return X.dot(weights)

eps=1e-5

def loss(X, Y, w, tau):
    pred = predict(X,w)
    err = pred - Y   # невязка (ошибка прогноза)
    # регуляризируем только w без последнего фиктивного признака
    return np.mean(err ** 2) + (tau / 2) * np.sum(w[:-1] ** 2)

def main(X, Y, lambda_reg : float, exp : float = 0.001, iter = 10000):
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
        prediction : float = np.dot(weights,X_i)
        error : float = float(prediction) - float(y_i)
        gradient = 2 * error * X_i
        gradient[1:]+= lambda_reg * weights[1:]
        weights -= (1 / (step + 1)) * gradient
        curr_loss = loss(X, Y, weights, lambda_reg)
        curr_Q = (curr_loss) * exp + (1 - exp) * Q
        mem_Q = np.append(mem_Q, curr_Q)
        if (abs(curr_Q - Q)  < eps):
            cnt += 1
            if cnt == 200:
                mem_weights = weights
                break
        else: 
            cnt = 0
        Q = curr_Q
        mem_weights = weights
    return mem_weights, mem_Q