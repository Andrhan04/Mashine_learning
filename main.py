import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# ----------- Импорт и подготовка данных -----------

iris = load_iris()

X = iris.data
Y = iris.target

# --------------------------------------------------

# ----------- Отрисовка данных -----------

def show_dataset(X, Y):
		for x_row, y_value in zip(X, Y):   # zip - обьединяет два списка (массива) в пары 
				print(f"{' '.join(map(str, x_row))} | {y_value}")   # 1) преобразуем в строки числа 2) превращаем в строку с разделителем пробелом


def plot_pairplot(dataset):
		df = pd.DataFrame(X, columns = dataset.feature_names)
		df["target"] = [dataset.target_names[i] for i in Y]
		print(df)
		
		sns.pairplot(df, hue = "target")
		plt.show()


def plot_loo_error(k_vals):
		plt.bar(range(1, len(k_vals) + 1), k_vals)
		plt.xlabel("k")
		plt.ylabel("LOO Accuracy")
		plt.title("LOO Error vs k")

		for i, val in enumerate(k_vals):   # получаем и индекс и значение 
				plt.text(i + 1, val, str(val), ha = "center", va = "bottom", fontsize = 8)

		plt.show()

# ----------------------------------------

# ------------------- Ядра ----------------------

def __epanechnikov_kernel(dist):
		if abs(dist) <= 1:
				return (3 / 4) * (1 - (dist * dist))
		else:
				return 0

# -----------------------------------------------


# ------------------- Функции -------------------

def __evc_distance(x, y):
		return np.sqrt(np.sum((x - y) ** 2))


def __search_label(obj):
		most_common_label = None
		max_cnt = 0
		for label, count in obj.items():
				if count > max_cnt:
						most_common_label = label
						max_cnt = count

		if most_common_label != None:
				return most_common_label
		else:
				print("Что-то не так")
				raise Exception("Метка не найдена")


def KNN(x_train, x_test, y_train, k):
		dist = [__evc_distance(x_test, x) for x in x_train]
		# print(dist)
		neighbors = np.argsort(dist)[:k]   # получение индексов элементов массива, отсортированных по возрастанию со срезом в k
		# print(neighbors)
		labels = [y_train[i] for i in neighbors]
		# print(labels)

		label_cnt = {}
		for label in labels:
				if label in label_cnt:
						label_cnt[label] += 1
				else:
						label_cnt[label] = 1
		
		return __search_label(label_cnt)


def KNN_with_weight(x_train, x_test, y_train, k):
		dist = [__evc_distance(x_test, x) for x in x_train]
		# print(dist)
		neighbors = np.argsort(dist)[:k]
		# weights = [__epanechnikov_kernel(d) for d in dist]
		weights = [__epanechnikov_kernel(d / __evc_distance(x_test, x_train[k + 1])) for d in dist]
		# print(weights) 

		class_weights = {}
		for i in neighbors:
				label = y_train[i]
				weight = weights[i]
				if label not in class_weights:
						class_weights[label] = 0
				class_weights[label] += weight

		return __search_label(class_weights)


# def LOO(X, Y):
# 		max_k = int(math.ceil(math.sqrt(len(X))))   # беру примерное значение k как корень всех элементов округленный в большую сторону
# 		k_accuracy = [0] * max_k

# 		for i in range(len(X)):
# 				x_test = X[i]
# 				x_train = np.delete(X, i, axis = 0)   # удаляем строку i 
# 				y_test = Y[i]
# 				y_train = np.delete(Y, i)

# 				for k in range(1, max_k + 1):
# 						y_pred = KNN(x_train, x_test, y_train, k)
# 						if y_pred == y_test:
# 								k_accuracy[k - 1] += 1

# 		return k_accuracy


def LOO(X, Y, k):
		errors = 0
		
		for i in range(len(X)):
				x_test = X[i]
				x_train = np.delete(X, i, axis = 0)
				y_test = Y[i]
				y_train = np.delete(Y, i)

				y_pred = KNN(x_train, x_test, y_train, k)
				if y_pred != y_test:
						errors += 1

		return errors

# ----------------------------------------------

# ----------------- MAIN ------------------------

def main():
		show_dataset(X, Y)

		plot_pairplot(iris)

		k_vals = []
		k_max = 100
		for k in range(1, k_max + 1):
				k_vals.append(LOO(X, Y, k))

		plot_loo_error(k_vals)

		k = k_vals.index(min(k_vals)) + 1
		print(f"Лучший k = {k}")

		x_my = np.array([5.1, 3.5, 1.4, 0.2])
		y_my = 0

		print(f"Тестовая точка - { x_my } | { y_my }")

		y_knn_1 = KNN(X, x_my, Y, k)
		print(f"KNN говорит: { x_my } это - { y_knn_1 }")
		y_knn_2 = KNN_with_weight(X, x_my, Y, k)
		print(f"KNN с весами говорит: { x_my } это - { y_knn_2 }")

		if y_knn_1 == y_my and y_knn_2 != y_my:
				print("KNN прав!")
		elif y_knn_2 == y_my and y_knn_1 != y_my:
				print("KNN прав!")
		elif y_knn_1 == y_my and y_knn_2 == y_my:
				print("Оба правы!")
		else:
				print("Увы!")
        
# -----------------------------------------------



# ------------------ Запуск ---------------------

if __name__ == "__main__":
		main()

# -----------------------------------------------