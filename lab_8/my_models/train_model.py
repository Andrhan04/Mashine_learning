import keras # type: ignore
import os
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import to_categorical
from images.painter import draw_accuracy, draw_dataset, draw_loss


model_name = "example"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Отключаем GPU, чтобы TensorFlow работал только на CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Скрываем предупреждения TensorFlow


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # Добавлям канал, то есть изображение черно-белое. Необходимо для Keras
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    # Разделяем на тренировочную и валидационную выборки  
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    # Преобразуем метки в one-hot
    y_test = to_categorical(y_test, 10)
    y_val = to_categorical(y_val, 10)
    y_train = to_categorical(y_train, 10)
    return X_train, y_train, X_test, y_test, X_val, y_val


def build_model(X_train, y_train, X_test, y_test):
    datagen = ImageDataGenerator(   # Аугментация данных
        rotation_range=10,          # вращение 10 градусов
        width_shift_range=0.1,      # горизонтальные сдвиги
        height_shift_range=0.1,     # вертикальные сдвиги
        zoom_range=0.1,             # увеличение/уменьшение
    )
    
    
    datagen.fit(X_train)
    
    model = models.Sequential([                                              # Последовательная модель 
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), # Сверточный слой
        layers.MaxPooling2D((2,2)),                                         # Пуллинговый слой
        layers.Conv2D(64, (3,3), activation='relu'),                        # Сверточный слой
        layers.MaxPooling2D((2,2)),                                         # Пуллинговый слой
        layers.Conv2D(64, (3,3), activation='relu'),                        # Сверточный слой
        layers.Flatten(),                                                   # Преобразует матрицу в массив
        layers.Dense(64, activation='relu'),                                # Полносвязанный слой
        layers.Dropout(0.5),                                                # в каждом батче случайно отключается 50% нейронов
        layers.Dense(10, activation='softmax')                              # Вероятности для цифр
    ]) 
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
    return model, history
    
def traning_model():
    X_train, y_train, X_test_ev, y_test_ev, X_test, y_test = load_data()
    model, history = build_model(X_train, y_train, X_test, y_test)
    model.save(f"my_models\\{model_name}.keras") # Сохранение модели
    test_loss, test_acc = model.evaluate(X_test_ev, y_test_ev) # Оценка обученной модели на тестовой выборке
    print(f'Точность на тестовой: {test_acc}, Потери на тестовой: {test_loss}')
    draw_dataset(X_train, y_train)
    draw_accuracy(history)
    draw_loss(history)

#traning_model()