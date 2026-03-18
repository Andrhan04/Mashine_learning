import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Sequence


# путь к csv файлу с погодными измерениями
CSV_PATH = Path("jena_climate_2009_2016.csv")

# сколько исходных шагов по времени смотрим назад
# в датасете один шаг это 10 минут
LOOKBACK = 720

# берём не каждую запись, а каждую шестую
# это значит что фактически используем один замер в час
STEP = 6

# насколько далеко в будущее предсказываем целевое значение
# 144 шага по 10 минут это 24 часа
DELAY = 144

# размер батча для обучения и проверки
BATCH_SIZE = 128

# правая граница обучающей части
TRAIN_MAX_INDEX = 200000

# правая граница валидационной части
VAL_MAX_INDEX = 300000


class ClimateSequence(Sequence):
    # этот класс хранит список допустимых строк
    # и по запросу собирает из них батчи
    # keras умеет работать с такими объектами напрямую
    def __init__(self, data, lookback, delay, min_index, max_index, batch_size, step, shuffle):
        self.data = data
        self.lookback = lookback
        self.delay = delay
        self.min_index = min_index
        self.max_index = max_index
        self.batch_size = batch_size
        self.step = step
        self.shuffle = shuffle

        # заранее строим список строк, которые можно использовать как центр примера
        self.row_indices = self._build_row_indices()

        # для train набора перемешиваем порядок строк сразу при создании
        if self.shuffle:
            np.random.shuffle(self.row_indices)

    def _build_row_indices(self):
        # если верхняя граница не задана, значит работаем до конца массива
        # но нужно оставить запас справа на delay
        if self.max_index is None:
            last_row = len(self.data) - self.delay - 1
        else:
            last_row = self.max_index

        # первая допустимая строка начинается только после lookback шагов
        first_row = self.min_index + self.lookback

        # столько строк можно использовать в этом поднаборе
        row_count = last_row - first_row

        # здесь будут номера строк, по которым можно строить примеры
        row_indices = np.zeros((row_count,), dtype=np.int32)

        current_row = first_row
        position = 0

        # заполняем массив индексов подряд
        while current_row < last_row:
            row_indices[position] = current_row
            current_row += 1
            position += 1

        return row_indices

    def __len__(self):
        # keras спрашивает длину последовательности в батчах
        return math.ceil(len(self.row_indices) / self.batch_size)

    def __getitem__(self, index):
        # переводим номер батча в границы по индексам строк
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.row_indices))

        # берём строки, из которых нужно собрать текущий батч
        rows = self.row_indices[start:end]

        # из выбранных строк строим входы и цели
        samples, targets = build_batch(self.data, rows, self.lookback, self.delay, self.step)
        return samples, targets

    def on_epoch_end(self):
        # после каждой эпохи можно заново перемешать train данные
        if self.shuffle:
            np.random.shuffle(self.row_indices)


def load_jena_csv(csv_path):
    # читаем весь файл в память как список строк
    with csv_path.open("r", encoding="utf-8") as file:
        lines = file.read().splitlines()

    # в первой строке лежат имена колонок
    header = lines[0].split(",")
    print("Колонки:", header[:5], "...", f"(всего {len(header)})")

    # первый столбец это дата и время, поэтому берём на один признак меньше
    data = np.zeros((len(lines) - 1, len(header) - 1), dtype=np.float32)

    line_index = 1

    # идём по всем строкам после заголовка и переводим значения в float
    while line_index < len(lines):
        values = lines[line_index].split(",")[1:]   # пропускаем дату и время
        data[line_index - 1, :] = np.array(values, dtype=np.float32)
        line_index += 1

    return data


def normalize_train_stats(data, train_max_index):
    # среднее считаем только по обучающей части
    mean = data[:train_max_index].mean(axis=0)

    # стандартное отклонение тоже считаем только по обучающей части
    std = data[:train_max_index].std(axis=0)

    # работаем с копией, чтобы не менять исходный массив
    normalized = data.copy()

    # стандартная нормализация признаков
    normalized -= mean
    normalized /= std

    return normalized, mean, std


def build_batch(data, rows, lookback, delay, step):
    # сколько временных точек попадёт в один пример после прореживания
    time_steps = lookback // step

    # сколько всего признаков в одном измерении
    feature_count = data.shape[-1]

    # сюда сложим входные последовательности
    samples = np.zeros((len(rows), time_steps, feature_count), dtype=np.float32)

    # сюда сложим целевые значения температуры
    targets = np.zeros((len(rows),), dtype=np.float32)

    row_position = 0

    # идём по всем строкам, которые должны попасть в батч
    while row_position < len(rows):
        row = rows[row_position]

        # начинаем смотреть в прошлое от текущей строки
        data_index = row - lookback
        time_position = 0

        # собираем историю с шагом step
        while data_index < row:
            samples[row_position, time_position] = data[data_index]
            data_index += step
            time_position += 1

        # целевое значение это температура через delay шагов
        # индекс 1 это второй числовой столбец в датасете
        targets[row_position] = data[row + delay][1]
        row_position += 1

    return samples, targets


def make_datasets(data):
    # train набор берётся из начала массива и перемешивается
    train_data = ClimateSequence(
        data=data,
        lookback=LOOKBACK,
        delay=DELAY,
        min_index=0,
        max_index=TRAIN_MAX_INDEX,
        batch_size=BATCH_SIZE,
        step=STEP,
        shuffle=True,
    )

    # val набор идёт сразу после train и не перемешивается
    val_data = ClimateSequence(
        data=data,
        lookback=LOOKBACK,
        delay=DELAY,
        min_index=TRAIN_MAX_INDEX + 1,
        max_index=VAL_MAX_INDEX,
        batch_size=BATCH_SIZE,
        step=STEP,
        shuffle=False,
    )

    # test набор это хвост данных после validation части
    test_data = ClimateSequence(
        data=data,
        lookback=LOOKBACK,
        delay=DELAY,
        min_index=VAL_MAX_INDEX + 1,
        max_index=None,
        batch_size=BATCH_SIZE,
        step=STEP,
        shuffle=False,
    )

    return train_data, val_data, test_data


def evaluate_naive_method(val_data, std):
    # здесь считаем качество очень простой базовой идеи
    # прогноз равен последнему известному значению температуры во входе
    maes = []
    batch_index = 0

    # проходим по всем батчам validation набора
    while batch_index < len(val_data):
        samples, targets = val_data[batch_index]
        predictions = samples[:, -1, 1]   # берём последнюю температуру из входной истории
        batch_mae = np.mean(np.abs(predictions - targets))
        maes.append(batch_mae)
        batch_index += 1

    # усредняем ошибку по всем батчам
    mean_mae = float(np.mean(maes))

    # возвращаем ошибку обратно в градусах, а не в нормализованном масштабе
    return mean_mae * std[1]


def plot_history(history, title_prefix):
    # история обучения хранит значение функции потерь по эпохам
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title(f"{title_prefix}: training and validation loss")
    plt.legend()
    plt.show()


def train_and_evaluate_model(model, train_data, val_data, test_data, epochs, title):
    # печатаем разделитель, чтобы модели было проще различать в консоли
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    model.summary()

    # обучаем модель на train и проверяем на validation после каждой эпохи
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        verbose=1,
    )

    # после обучения отдельно оцениваем качество на test части
    print("\nTest evaluation:")
    test_metrics = model.evaluate(test_data, verbose=1)
    print("Test metrics:", test_metrics)

    plot_history(history, title)
    return history


def build_dense_model(num_features):
    # сначала разворачиваем временную последовательность в один длинный вектор
    # затем подаём его в обычные полносвязные слои
    model = Sequential([
        layers.Input(shape=(LOOKBACK // STEP, num_features)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])

    # mae удобна здесь, потому что мы решаем задачу регрессии
    model.compile(optimizer=RMSprop(), loss="mae")
    return model


def build_gru_model(num_features):
    # базовая рекуррентная модель
    # GRU сама обрабатывает последовательность по времени
    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.GRU(32),
        layers.Dense(1),
    ])

    model.compile(optimizer=RMSprop(), loss="mae")
    return model


def build_gru_dropout_model(num_features):
    # эта версия похожа на предыдущую
    # но в ней добавлена регуляризация через dropout
    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.GRU(
            32,
            dropout=0.2,
            recurrent_dropout=0.2,
        ),
        layers.Dense(1),
    ])

    model.compile(optimizer=RMSprop(), loss="mae")
    return model


def build_stacked_gru_model(num_features):
    # здесь первая GRU возвращает всю последовательность скрытых состояний
    # это нужно, чтобы вторая GRU тоже получила временной ряд на вход
    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.GRU(
            32,
            dropout=0.1,
            recurrent_dropout=0.5,
            return_sequences=True,
        ),
        layers.GRU(
            64,
            activation="relu",
            dropout=0.1,
            recurrent_dropout=0.5,
        ),
        layers.Dense(1),
    ])

    model.compile(optimizer=RMSprop(), loss="mae")
    return model


def build_bidirectional_gru_model(num_features):
    # эта модель читает последовательность сразу в двух направлениях
    # для временных рядов такой подход не всегда полезен, но сравнить интересно
    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(1),
    ])

    model.compile(optimizer=RMSprop(), loss="mae")
    return model


def run_model(model_builder, model_name, epochs, num_features, data):
    # для каждой модели заново создаём наборы данных
    # это удобно и убирает зависимость от внутреннего состояния после прошлой эпохи
    train_data, val_data, test_data = make_datasets(data)

    # строим конкретную модель по переданной функции
    model = model_builder(num_features)

    train_and_evaluate_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=epochs,
        title=model_name,
    )


def main():
    # фиксируем случайность, чтобы результаты были более повторяемыми
    np.random.seed(42)
    tf.random.set_seed(42)

    # загружаем исходные измерения
    raw_data = load_jena_csv(CSV_PATH)

    # нормализуем признаки по статистике train части
    normalized_data, mean, std = normalize_train_stats(raw_data, TRAIN_MAX_INDEX)

    print("float_data shape:", normalized_data.shape)
    print("num_features:", normalized_data.shape[-1])

    # для наивной оценки нам нужен только validation набор
    _, val_data, _ = make_datasets(normalized_data)
    naive_mae_celsius = evaluate_naive_method(val_data, std)

    print("\n" + "=" * 80)
    print("Baseline без машинного обучения")
    print("=" * 80)
    print(f"Validation MAE (в градусах): {naive_mae_celsius:.3f}")

    # число признаков нужно всем моделям при создании входного слоя
    num_features = normalized_data.shape[-1]

    # дальше по очереди обучаем и сравниваем несколько архитектур
    run_model(build_dense_model, "Dense baseline", 20, num_features, normalized_data)
    run_model(build_gru_model, "Simple GRU", 20, num_features, normalized_data)
    run_model(build_gru_dropout_model, "GRU with dropout + recurrent_dropout", 40, num_features, normalized_data)
    run_model(build_stacked_gru_model, "Stacked GRU", 40, num_features, normalized_data)
    run_model(build_bidirectional_gru_model, "Bidirectional GRU", 40, num_features, normalized_data)


if __name__ == "__main__":
    main()
