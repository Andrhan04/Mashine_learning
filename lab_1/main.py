import numpy as np
import Class.Paint_data as painter
import Class.load_data as loader
import Class.Classification as cls
import Class.generalizing_ability as Validation

X,Y,iris  =  loader.give_iris()
painter.show_dataset(X, Y)

painter.plot_pairplot(iris, X, Y)

k_vals = []
k_max = 100
for k in range(1, k_max + 1):
        k_vals.append(Validation.LOO(X, Y, k))

painter.plot_loo_error(k_vals)

k = k_vals.index(min(k_vals)) + 1
print(f"Лучший k = {k}")

x_my = np.array([5.1, 3.5, 1.4, 0.2])
y_my = 0

print(f"Тестовая точка - { x_my } | { y_my }")

y_knn_1 = cls.KNN(X, x_my, Y, k)
print(f"KNN говорит: { x_my } это - { y_knn_1 }")
y_knn_2 = cls.KNN_with_weight(X, x_my, Y, k)
print(f"KNN с весами говорит: { x_my } это - { y_knn_2 }")

if y_knn_1 == y_my and y_knn_2 != y_my:
        print("KNN прав!")
elif y_knn_2 == y_my and y_knn_1 != y_my:
        print("KNN c весами прав!")
elif y_knn_1 == y_my and y_knn_2 == y_my:
        print("Оба правы!")
else:
        print("Увы!")