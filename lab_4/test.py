import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



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


def evaluate_models(X_train, X_test, Y_train, Y_test, C_arr):
    # обучает и тестирует модели логистической регрессии и линейного svm для каждого значения C
    log_acc, svc_acc = [], []
    log_coefs, svc_coefs = [], []

    for C in C_arr:
        log = LogisticRegression(C = C, max_iter = 10000).fit(X_train, Y_train)
        svc = LinearSVC(C = C, max_iter = 10000).fit(X_train, Y_train)

        # Внутри fit
        # Проверяет входные данные (validate_data)
        # Выбирает решатель (solver) и формирует задачу оптимизации
        # Вычисляет classes_
        # Ищет минимум функции потерь: логистическая регрессия + L2 регуляризация (C управляет штрафом)
        # Если solver='lbfgs', то градиентный метод второго порядка (Гессиан)
        # Если liblinear, то решается как SVM, только с лог-лоссом (функция потерь для бинарной классификации, основанная на правдоподобии)
        # Итеративно обновляет коэффициенты, пока не сойдётся
        # В результате model.coef_, model.intercept_ и classes_

        log_acc.append(accuracy_score(Y_test, log.predict(X_test)))
        svc_acc.append(accuracy_score(Y_test, svc.predict(X_test)))

        # Внутри predict 
        # Счёт уверенности (scores = self.decision_function(X))
        # Определение предсказанных индексов классов
        # Преобразование индексов в реальные метки

        log_weights = log.coef_[0]
        svc_weights = svc.coef_[0]

        log_coefs.append(log_weights)
        svc_coefs.append(svc_weights)

    return log_acc, svc_acc, log_coefs, svc_coefs


def plot_accuracy(C_arr, log_acc, svc_acc):
    # строит график зависимости accuracy от параметра регуляризации C
    plt.plot(C_arr, log_acc, label = 'LogReg', marker = 'o')
    plt.plot(C_arr, svc_acc, label = 'LinSVC', marker = 'x')

    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Точность при C')
    plt.legend()
    plt.show()


def plot_coeffs(C_arr, coefs, title):
    # График показывает, как значения коэффициентов модели изменяются при различной силе регуляризации
    # for i, C in enumerate(C_arr):
    #     plt.plot(coefs[i], label = f'C = {C}')

    coefs = np.array(coefs)
    for i in range(coefs.shape[1]):
        plt.semilogx(C_arr, coefs[:, i], label = f'Feature {i}')
    
    plt.title(f'{title} Значение на С')
    plt.xlabel('C')
    plt.ylabel('Коэффициенты (веса признаков)')
    # plt.xlabel('Feature index')
    # plt.ylabel('Coeff value')
    # plt.legend(ncol = 2, fontsize = 'small')
    plt.tight_layout()   # вычисляет оптимальные значения отступов между элементами
    plt.show()



def main():
    X_train, X_test, Y_train, Y_test = load_and_split_data()
    X_train, X_test = scale_data(X_train, X_test)

    C_arr = [ 0.0000000001, 0.000001, 0.0001, 0.001, 0.01, 0.1,  1, 10, 100, 1000, 10000, 100000, 100000000 ]

    log_acc, svc_acc, log_coefs, svc_coefs = evaluate_models(
        X_train, X_test, Y_train, Y_test, C_arr
    )

    plot_accuracy(C_arr, log_acc, svc_acc)
    plot_coeffs(C_arr, log_coefs, 'LogReg')
    plot_coeffs(C_arr, svc_coefs, 'LinSVC')

    print("Logistic Regression:")
    best_idx = np.argmax(log_acc)
    print(f"  Best C: {C_arr[best_idx]}")
    print(f"  Balanced Accuracy: {log_acc[best_idx]}")

    print("Linear SVM:")
    best_idx = np.argmax(svc_acc)
    print(f"  Best C: {C_arr[best_idx]}")
    print(f"  Balanced Accuracy: {svc_acc[best_idx]}")


if __name__ == "__main__":
    main()
