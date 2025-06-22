from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def model(X_train, X_test, Y_train, Y_test, C_arr):
    # обучает и тестирует модели логистической регрессии и линейного svm для каждого значения C
    acc = []
    coef = []

    for C in C_arr:
        log = LogisticRegression(C = C, max_iter = 10000).fit(X_train, Y_train)

        # Внутри fit
        # Проверяет входные данные (validate_data)
        # Выбирает решатель (solver) и формирует задачу оптимизации
        # Вычисляет classes_
        # Ищет минимум функции потерь: логистическая регрессия + L2 регуляризация (C управляет штрафом)
        # Если solver='lbfgs', то градиентный метод второго порядка (Гессиан)
        # Если liblinear, то решается как SVM, только с лог-лоссом (функция потерь для бинарной классификации, основанная на правдоподобии)
        # Итеративно обновляет коэффициенты, пока не сойдётся
        # В результате model.coef_, model.intercept_ и classes_

        acc.append(accuracy_score(Y_test, log.predict(X_test)))

        # Внутри predict 
        # Счёт уверенности (scores = self.decision_function(X))
        # Определение предсказанных индексов классов
        # Преобразование индексов в реальные метки

        log_weights = log.coef_[0]

        coef.append(log_weights)

    return acc, coef