import matplotlib.pyplot as plt
import keras
import numpy as np

def draw_dataset(X_train, y_train):
    # Как выглядит датасет
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([]) # Убираем деления по ОХ
        plt.yticks([]) # Убираем деления по ОУ
        plt.grid(False)
        plt.imshow(X_train[i].reshape(28, 28), 'gray')
        plt.xlabel(y_train[i])
    plt.savefig('images\\dataset')
    plt.show()
    
def draw_accuracy(history):
    # Точность
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Точность обучения')
    plt.legend()
    plt.savefig('images\\accuratly')
    plt.show()

def draw_loss(history):
    # Потери
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Потери обучения')
    plt.legend()
    plt.savefig('images\\loss')
    plt.show()





from PIL import Image
import os
def save_images():
    # Загружаем датасет
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Создаем папку для сохранения изображений
    os.makedirs('digits_images', exist_ok=True)

    # Находим по одному примеру для каждой цифры (0-9)
    for digit in range(10):
        # Находим первый попавшийся индекс для текущей цифры
        indices = np.where(y_train == digit)[0]
        idx = indices[0]
        
        # Получаем изображение
        image = X_train[idx]
        
        # Сохраняем как PNG файл
        image_path = f'images/num_{digit}.png'
        
        # Используем PIL для сохранения
        img = Image.fromarray(image.astype('uint8'))
        img.save(image_path)
        
        print(f'Сохранено: {image_path}')
