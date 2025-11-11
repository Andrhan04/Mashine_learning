import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_color_dataset():
    """Загрузка датасета из файла"""
    try:
        with np.load('color_dataset.npz') as data:
            X = data['data']
            y = data['target']
        
        print(f"Загружено из color_dataset.npz")
        print(f"Размер data: {X.shape}, размер target: {y.shape}")
        print(f"Класс 0 (холодные): {np.sum(y == 0)}, класс 1 (теплые): {np.sum(y == 1)}")
        
        return X, y
    except FileNotFoundError:
        print("Ошибка: Файл color_dataset.npz не найден!")
        print("Сначала запустите код генерации данных")
        return None, None

def create_model():
    """Создание нейронной сети"""
    # model = Sequential([
    #     Dense(64, activation='relu', input_shape=(3,)),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    # model = Sequential([
    #     Dense(16, activation='relu', input_shape=(3,)),
    #     Dense(8, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    model = Sequential([
        Dense(8, activation='relu', input_shape=(3,)),  # Всего один скрытый слой
        Dense(1, activation='sigmoid')                   # Выходной слой
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Загрузка данных
    print("Загрузка датасета из color_dataset.npz...")
    data, target = load_color_dataset()
    
    if data is None:
        exit(1)
    
    # Преобразование target в одномерный массив
    if len(target.shape) > 1:
        target = target.flatten()
        print(f"Преобразована форма target: {target.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nОбучающая выборка: {X_train_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    # Создание и обучение модели
    print("\nСоздание модели...")
    model = create_model()
    model.summary()
    
    print("\nОбучение модели...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Оценка модели
    print("\nОценка на тестовых данных...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Тестовая точность: {test_accuracy:.2%}")
    print(f"Тестовые потери: {test_loss:.4f}")
    
    # Визуализация
    plot_training_history(history)
    
    # Сохранение модели и scaler
    model.save('color_classifier_model.h5')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    
    print("\nМодель сохранена как 'color_classifier_model.h5'")
    print("Параметры масштабирования сохранены")