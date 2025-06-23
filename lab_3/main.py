import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_california_housing
from Class.SGD import main as sgd
from Class.generalizing_ability import Cross_validation, r2_score

Data = fetch_california_housing()
X, Y = Data.data, Data.target   
#-------------------------------------------------------------------------------
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
Y = (Y - np.mean(Y)) / np.std(Y)
X=np.hstack((-np.ones((X.shape[0],1)),X))
#------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

lambda_reg = np.arange(0.01, 1, 0.05)
r2_arr = []
for tau in lambda_reg:
    w, curr_Q = sgd(X_train, y_train, tau)
    y_pred = X_test.dot(w)
    #r2 = r2_score(y_test, y_pred)
    r2 = (np.corrcoef(X_test.dot(w),y_test)[0,1]) ** 2
    r2_arr.append(r2)

best_lambda = lambda_reg[np.argmax(r2_arr)]
weights, best_Q =  sgd(X_train, y_train, best_lambda)

plt.figure(figsize = (10, 5))
plt.plot(lambda_reg, r2_arr, marker = 'o')
plt.title("Зависимость $R^2$ от $\\lambda$")
plt.xlabel("$\\lambda$")
plt.ylabel("$R^2$")
plt.grid()
plt.savefig('lab_3\\images\\R.png')
plt.show()

loss = Cross_validation(X_test, y_test, weights, best_lambda)

r=np.corrcoef(X_test.dot(weights),y_test)[0,1]
best_coef = r**2

plt.figure(figsize = (10, 5))
plt.plot(best_Q)
plt.title("График сходимости Q")
plt.xlabel("Итерация")
plt.ylabel("Q")
plt.grid()
plt.savefig('lab_3\\images\\q.png')
plt.show()

print("функция потерь:",loss)
print("коэф. корр.",best_coef)
print("регуляризатор",best_lambda)
#print(weights)