import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_trained_model():
    """Загрузка обученной модели и scaler"""
    try:
        model = load_model('color_classifier_model.h5')
        scaler_mean = np.load('scaler_mean.npy')
        scaler_scale = np.load('scaler_scale.npy')
        
        # Создаем scaler с загруженными параметрами
        scaler = StandardScaler()
        scaler.mean_ = scaler_mean
        scaler.scale_ = scaler_scale
        
        print("Модель и scaler успешно загружены!")
        return model, scaler
        
    except FileNotFoundError as e:
        print(f"Ошибка загрузки: {e}")
        print("Сначала обучите модель с помощью train_model.py")
        return None, None

def predict_single_color(model, scaler, r, g, b):
    """Предсказание класса для одного цвета"""
    # Масштабирование входных данных
    rgb_scaled = scaler.transform([[r, g, b]])
    
    # Предсказание
    prediction = model.predict(rgb_scaled, verbose=0)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    return predicted_class, confidence, prediction

def display_color_prediction(r, g, b, predicted_class, confidence, raw_prediction):
    """Красивый вывод результата предсказания"""
    color_name = 'Теплый' if predicted_class == 1 else 'Холодный'
    color_emoji = '🔴' if predicted_class == 1 else '🔵'
    
    print(f"\n{color_emoji} Цвет RGB({r}, {g}, {b})")
    print(f"   Класс: {color_name}")
    print(f"   Уверенность: {confidence:.2%}")
    print(f"   Сырое значение: {raw_prediction:.4f}")
    print("-" * 40)

def interactive_mode(model, scaler):
    """Интерактивный режим для ввода цветов"""
    print("\n🎨 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Вводите RGB значения (0-255) через пробел")
    print("Пример: 255 0 0")
    print("Для выхода введите 'q'")
    
    while True:
        try:
            user_input = input("\nВведите RGB: ").strip()
            
            if user_input.lower() == 'q':
                print("Выход из интерактивного режима")
                break
                
            rgb_values = list(map(int, user_input.split()))
            
            if len(rgb_values) != 3:
                print("Ошибка: нужно ввести 3 числа (R G B)")
                continue
                
            r, g, b = rgb_values
            
            if not all(0 <= x <= 255 for x in [r, g, b]):
                print("Ошибка: значения должны быть в диапазоне 0-255")
                continue
                
            predicted_class, confidence, raw_prediction = predict_single_color(model, scaler, r, g, b)
            display_color_prediction(r, g, b, predicted_class, confidence, raw_prediction)
            
        except ValueError:
            print("Ошибка: введите целые числа через пробел")
        except KeyboardInterrupt:
            print("\nВыход из интерактивного режима")
            break

def batch_predict_colors(model, scaler, color_list):
    """Пакетное предсказание для списка цветов"""
    print("\n📊 ПАКЕТНОЕ ПРЕДСКАЗАНИЕ")
    print("=" * 50)
    
    for i, (r, g, b) in enumerate(color_list, 1):
        predicted_class, confidence, raw_prediction = predict_single_color(model, scaler, r, g, b)
        print(f"{i:2d}. ", end="")
        display_color_prediction(r, g, b, predicted_class, confidence, raw_prediction)

def main():
    """Основная функция"""
    print("🎨 Классификатор цветов (Теплые/Холодные)")
    print("=" * 50)
    
    # Загрузка модели
    model, scaler = load_trained_model()
    if model is None:
        return
    
    # Тестовые примеры
    test_colors = [
        (255, 0, 0),      # Красный - теплый
        (0, 0, 255),      # Синий - холодный
        (255, 165, 0),    # Оранжевый - теплый
        (0, 128, 128),    # Бирюзовый - холодный
        (255, 192, 203),  # Розовый - теплый
        (70, 130, 180),   # Стальной синий - холодный
    ]
    
    # Пакетное предсказание
    batch_predict_colors(model, scaler, test_colors)
    
    # Интерактивный режим
    interactive_mode(model, scaler)

if __name__ == "__main__":
    main()