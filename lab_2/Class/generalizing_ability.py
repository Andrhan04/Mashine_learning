import numpy as np
import Class.Classification as cls

def LOO(X, Y, k):
		errors = 0
		for i in range(len(X)):
				x_test = X[i]
				x_train = np.delete(X, i, axis = 0)
				y_test = Y[i]
				y_train = np.delete(Y, i)

				y_pred = cls.KNN(x_train, x_test, y_train, k)
				if y_pred != y_test:
						errors += 1
		return errors