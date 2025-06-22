from Class.get_data import Get_standart_data
from Class.Logistic import model as LogisticRegression
from Class.SVM import model as SVM
from Class.painter import plot_accuracy, plot_coeffs
import numpy as np

X_train, X_test, Y_train, Y_test = Get_standart_data()
C_arr = [ 0.0000000001, 0.000001, 0.0001, 0.001, 0.01, 0.1,  1, 10, 100, 1000, 10000, 100000, 100000000 ]
log_acc, log_coefs = LogisticRegression(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test, C_arr = C_arr)
svc_acc, svc_coefs = SVM(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test, C_arr = C_arr)

plot_accuracy(C_arr, log_acc, svc_acc)
plot_coeffs(C_arr, log_coefs, 'LogReg')
plot_coeffs(C_arr, svc_coefs, 'SVM')

print("Logistic Regression:")
best_idx = np.argmax(log_acc)
print(f"  Best C: {C_arr[best_idx]}")
print(f"  Balanced Accuracy: {log_acc[best_idx]}")

print("Linear SVM:")
best_idx = np.argmax(svc_acc)
print(f"  Best C: {C_arr[best_idx]}")
print(f"  Balanced Accuracy: {svc_acc[best_idx]}")