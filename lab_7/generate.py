import numpy as np
import matplotlib.pyplot as plt

class PointGenerator:
    def __init__(self, x_range=(-15, 15), y_range=(-10, 10)):
        self.x_range = x_range
        self.y_range = y_range
        
    def generate_points(self, n_points=1000):
        """Генерирует случайные точки в заданном диапазоне"""
        x = np.random.uniform(self.x_range[0], self.x_range[1], n_points)
        y = np.random.uniform(self.y_range[0], self.y_range[1], n_points)
        return x, y
    
    def classify_point(self, x, y):
        """
        Классифицирует точку:
        1 класс - в круге (x-5)^2 + y^2 = 100, но не в круге x^2 + y^2 = 100
        0 класс - остальные точки
        """
        # Проверяем принадлежность к кругам
        in_circle1 = (x-5)**2 + y**2 <= 100  # (x-5)^2 + y^2 = 100
        in_circle2 = x**2 + y**2 <= 100      # x^2 + y^2 = 100
        
        # Класс 1: в первом круге, но не во втором
        if (in_circle1 and not in_circle2) or (not in_circle1 and in_circle2):
            return 1
        else:
            return 0
    
    def generate_classified_data(self, n_points=1000):
        """Генерирует точки с метками классов"""
        x, y = self.generate_points(n_points)
        labels = np.array([self.classify_point(xi, yi) for xi, yi in zip(x, y)])
        return x, y, labels

class CircleVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
    def plot_circles(self):
        """Рисует границы окружностей"""
        # Первая окружность: (x-5)^2 + y^2 = 100
        theta = np.linspace(0, 2*np.pi, 100)
        x1 = 5 + 10 * np.cos(theta)  # радиус = 10
        y1 = 10 * np.sin(theta)
        
        # Вторая окружность: x^2 + y^2 = 100
        x2 = 10 * np.cos(theta)
        y2 = 10 * np.sin(theta)
        
        self.ax.plot(x1, y1, 'k--', linewidth=2, label='$(x-5)^2 + y^2 = 100$')
        self.ax.plot(x2, y2, 'k--', linewidth=2, label='$x^2 + y^2 = 100$')
        
    def plot_points(self, x, y, labels):
        """Рисует точки с цветами согласно классам"""
        class1_x = x[labels == 1]
        class1_y = y[labels == 1]
        class0_x = x[labels == 0]
        class0_y = y[labels == 0]
        
        self.ax.scatter(class1_x, class1_y, c='red', s=20, alpha=0.7, 
                       label='Класс 1 (в одном из 2-х кругов, но не в другом)')
        self.ax.scatter(class0_x, class0_y, c='blue', s=20, alpha=0.3, 
                       label='Класс 0 (остальные)')
        
    def show_plot(self, title="Классификация точек по принадлежности к окружностям"):
        """Отображает график"""
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

def main():
    # Создаем генератор точек
    generator = PointGenerator()
    
    # Генерируем данные
    print("Генерация точек...")
    x, y, labels = generator.generate_classified_data(2000)
    
    # Анализируем результаты
    n_class1 = np.sum(labels == 1)
    n_class0 = np.sum(labels == 0)
    
    print(f"Всего точек: {len(labels)}")
    print(f"Класс 1: {n_class1} точек")
    print(f"Класс 0: {n_class0} точек")
    print(f"Процент класса 1: {n_class1/len(labels)*100:.2f}%")
    
    # Визуализируем
    visualizer = CircleVisualizer()
    visualizer.plot_circles()
    visualizer.plot_points(x, y, labels)
    visualizer.show_plot()
    
    # Пример проверки конкретных точек
    print("\nПримеры классификации конкретных точек:")
    test_points = [
        (10, 0),   # Должна быть класс 1 (только в красном круге)
        (0, 0),    # Должна быть класс 0 (в обоих кругах)
        (-8, 0),   # Должна быть класс 0 (вне обоих кругов)
        (12, 0),   # Должна быть класс 0 (вне обоих кругов)
    ]
    
    for point in test_points:
        x_pt, y_pt = point
        label = generator.classify_point(x_pt, y_pt)
        print(f"Точка ({x_pt}, {y_pt}) -> Класс {label}")

if __name__ == "__main__":
    main()