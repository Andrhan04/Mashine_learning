import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_simple_model():
    """Создание упрощенной нейронной сети с одним скрытым слоем"""
    model = Sequential([
        Dense(8, activation='relu', input_shape=(3,)),  # Один скрытый слой
        Dense(1, activation='sigmoid')                   # Выходной слой
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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
        return None, None

def evaluate_model_with_f1(model, X_test, y_test):
    """Оценка модели с вычислением F1-score и других метрик"""
    # Предсказание вероятностей
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Преобразование в бинарные предсказания (порог 0.5)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Вычисление F1-score
    f1 = f1_score(y_test, y_pred)
    
    # Дополнительные метрики
    accuracy = np.mean(y_pred == y_test)
    
    print("\n" + "="*50)
    print("📊 ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*50)
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Подробный отчет классификации
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Холодные', 'Теплые']))
    
    # Матрица ошибок
    print("🎯 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"   Предсказано →")
    print(f"    Холодные  Теплые")
    print(f"Холодные  {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Теплые    {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    return f1, y_pred

def plot_predictions_comparison(y_true, y_pred, X_test_original):
    """Визуализация сравнения предсказаний с истинными значениями"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # График правильных и неправильных предсказаний
    correct = y_pred == y_true
    incorrect = ~correct
    
    axes[0].scatter(range(len(y_true)), y_true, c=correct, 
                   cmap='coolwarm', alpha=0.6, s=50)
    axes[0].set_title('Правильные (синие) vs Неправильные (красные) предсказания')
    axes[0].set_xlabel('Номер примера')
    axes[0].set_ylabel('Класс (0=Холодный, 1=Теплый)')
    axes[0].set_yticks([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Распределение уверенности модели для каждого класса
    warm_colors = X_test_original[y_true == 1]
    cold_colors = X_test_original[y_true == 0]
    
    if len(warm_colors) > 0:
        axes[1].scatter(warm_colors[:, 0], warm_colors[:, 1], 
                       c='red', alpha=0.6, label='Теплые', s=50)
    if len(cold_colors) > 0:
        axes[1].scatter(cold_colors[:, 0], cold_colors[:, 1], 
                       c='blue', alpha=0.6, label='Холодные', s=50)
    
    axes[1].set_title('Распределение цветов в тестовой выборке')
    axes[1].set_xlabel('R компонент')
    axes[1].set_ylabel('G компонент')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Основная программа ---
if __name__ == "__main__":
    # Загрузка данных
    print("Загрузка датасета...")
    data, target = load_color_dataset()
    
    if data is None:
        exit(1)
    
    # Преобразование target в одномерный массив
    if len(target.shape) > 1:
        target = target.flatten()
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Сохраняем оригинальные тестовые данные для визуализации
    X_test_original = X_test.copy()
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nОбучающая выборка: {X_train_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    # Создание и обучение упрощенной модели
    print("\nСоздание упрощенной модели...")
    model = create_simple_model()
    model.summary()
    
    print("\nОбучение модели...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # Оценка модели с F1-score
    f1, y_pred = evaluate_model_with_f1(model, X_test_scaled, y_test)
    
    # Интерпретация F1-score
    print(f"\n💡 ИНТЕРПРЕТАЦИЯ F1-SCORE: {f1:.4f}")
    if f1 >= 0.9:
        print("   Отличный результат! 🎉")
    elif f1 >= 0.8:
        print("   Очень хороший результат! 👍")
    elif f1 >= 0.7:
        print("   Хороший результат ✅")
    elif f1 >= 0.6:
        print("   Удовлетворительный результат ⚠️")
    else:
        print("   Требует улучшения ❌")
    
    # Визуализация результатов
    plot_predictions_comparison(y_test, y_pred, X_test_original)
    
    # Примеры предсказаний с вероятностями
    print("\n🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
    print("-" * 40)
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        rgb = X_test_original[idx]
        true_label = y_test[idx]
        pred_proba = model.predict(X_test_scaled[idx:idx+1], verbose=0)[0][0]
        pred_label = 1 if pred_proba > 0.5 else 0
        
        status = "✓" if pred_label == true_label else "✗"
        print(f"{status} RGB{rgb} -> Истина: {'Теплый' if true_label else 'Холодный'}")
        print(f"   Предсказание: {'Теплый' if pred_label else 'Холодный'} ({pred_proba:.2%})")
        print()
    
    # Сохранение модели
    model.save('color_classifier_model.h5')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    print(f"\n💾 Упрощенная модель сохранена как 'color_classifier_model.h5'")
    print(f"F1-score модели: {f1:.4f}")