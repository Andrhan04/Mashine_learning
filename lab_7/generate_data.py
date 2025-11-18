import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from generate import PointGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


tf.config.set_visible_devices([], 'GPU')


def generate_ring(size_dataset=1000, test_size=0.2):
    generator = PointGenerator()
    x, y, labels = generator.generate_classified_data(size_dataset)
    
    # Правильно объединяем признаки - транспонируем, чтобы каждая строка была точкой (x, y)
    X_total = np.column_stack([x, y])
    Y_total = labels
    
    # Делим на train / test
    return train_test_split(X_total, Y_total, test_size=test_size, random_state=42)

<<<<<<< HEAD:lab_7/main.py


def build_model():
		model = models.Sequential([
				layers.Input(shape=(2,)),   # вход вектор длины 2 - (x, y)
				layers.Dense(25, activation='relu'),
				layers.Dense(1, activation='sigmoid')
		])
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model

def draw(data, label, X_train, Y_train):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='coolwarm', s=20, alpha=0.3)
	ax.scatter(data[:, 0], data[:, 1], c=label, cmap='coolwarm', s=120, alpha=0.3, marker='x')
	ax.set_xlim(-20, 20)
	ax.set_ylim(-15, 15)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title("Классификация точек по принадлежности к окружностям")
	ax.grid(True, alpha=0.3)
	ax.legend()
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.show()


def main():
		X_train, X_test, Y_train, Y_test = generate_ring()
=======
# --- параметры ---
N = 500
np.random.seed(42)

def rgb_to_lab_hue(r, g, b):
    rgb = np.array([[[r, g, b]]])  # Создание 3D тензора для skimage
    lab = color.rgb2lab(rgb)       # Конвертация RGB → LAB
    a, b_ = lab[0, 0, 1], lab[0, 0, 2]  # Извлечение компонент a и b
    h = np.degrees(np.arctan2(b_, a)) % 360  # Вычисление угла (hue)
    return h

def label_by_hue(h):
    if h <= 90 or h >= 330:  # Красные/желтые тона
        return 1             # Теплые цвета
    elif 150 <= h <= 270:    # Синие тона  
        return 0             # Холодные цвета
    else:                    # Зеленые/пурпурные
        return None          # Нейтральные - исключаются

def generate_color():
	data_list = []
	target_list = []
>>>>>>> f48ff3b46836a00d1fb2a850678fc1eb8d39d891:lab_7/generate_data.py

		model = build_model()
		model.fit(X_train, Y_train, epochs=200, verbose=0)

		generator = PointGenerator()
		x_test, y_test = generator.generate_points(1000)
		test_points = np.column_stack([x_test, y_test])
		preds = model.predict(test_points) > 0.5

		print("\nТестовые точки и предсказания:")
		for i in range(len(test_points)):
				x = test_points[i][0]
				y = test_points[i][1]
				pred = preds[i]
				print(i, ":", "(", x, ",", y, ")", "-> класс", int(pred))

		loss, acc = model.evaluate(X_test, Y_test, verbose=0)
		print(f"Точность модели на тесте: {acc}")   # точность
		print(f"Потери модели на тесте: {loss}")   # уверенность модели

		draw(X_test, Y_test,X_train ,Y_train)
		

<<<<<<< HEAD:lab_7/main.py
	

if __name__ == "__main__":
		main()
=======
generate_color()
# Что такое LAB:
# L — Lightness (яркость), от 0 (чёрный) до 100 (белый)
# a — ось от зелёного (–a) к красному (+a)
# b — ось от синего (–b) к жёлтому (+b)
>>>>>>> f48ff3b46836a00d1fb2a850678fc1eb8d39d891:lab_7/generate_data.py
