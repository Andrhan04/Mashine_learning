import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_california_housing
from Class.SGD import main as sgd
from Class.generalizing_ability import Cross_validation
Data = fetch_california_housing()
X, Y = Data.data, Data.target   
#-------------------------------------------------------------------------------
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # нормировка
Y = (Y - np.mean(Y)) / np.std(Y)
X=np.hstack((np.ones((X.shape[0],1)),X))
#------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

lambda_reg = np.arange(0.01, 1, 0.05)
best_loss : float = float('inf')
weights = None

for i in lambda_reg:
    current_weights = sgd(X_train, y_train, i)
    loss_time = Cross_validation(X_train, y_train, current_weights, i)
    if (loss_time < best_loss):
        weights = current_weights
        best_loss = loss_time
        best_lambda = i
        
loss = Cross_validation(X_test, y_test, weights, best_lambda)

r=np.corrcoef(X_test.dot(weights),y_test)[0,1]

print("функция потерь:",loss)
best_coef=r**2
print("коэф. корр.",best_coef)
print("регуляризатор",best_lambda)