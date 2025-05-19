import numpy as np
import Class.kernel as kern

def find_label(obj : dict):
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

def KNN(x_train : list, x_test : list, y_train : list, k : int):
		dist : list = [kern.evc_distance(x_test, x) for x in x_train]
		# print(dist)
		neighbors = np.argsort(dist)[:k]   # получение индексов элементов массива, отсортированных по возрастанию со срезом в k
		# print(neighbors)
		labels = [y_train[i] for i in neighbors]
		# print(labels)

		label_cnt : dict = {}
		for label in labels:
				if label in label_cnt:
						label_cnt[label] += 1
				else:
						label_cnt[label] = 1
		
		return find_label(label_cnt)


def KNN_with_weight(x_train : list, x_test : list, y_train : list, k : int):
		dist : list = [kern.evc_distance(x_test, x) for x in x_train]
		# print(dist)
		neighbors : list = np.argsort(dist)[:k]
		# weights = [kern.epanechnikov_kernel(d) for d in dist]
		weights = [kern.epanechnikov_kernel(d / kern.evc_distance(x_test, x_train[k + 1])) for d in dist]
		# print(weights) 

		class_weights : dict = {}
		for i in neighbors:
				label = y_train[i]
				weight = weights[i]
				if label not in class_weights:
						class_weights[label] = 0
				class_weights[label] += weight

		return find_label(class_weights)