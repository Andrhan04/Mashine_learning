"""Лабораторная работа №10.

Преобразование Фурье и сингулярно-спектральное разложение (SSA)
пульсового сигнала.

Задача:
    Для каждого пульсового сигнала из выборки определить частоту главной
    компоненты Фурье-разложения и частоту основной гармоники SSA, после чего
    сравнить их на общем графике.

Вход:
    Выборка пульсовых сигналов в каталоге data. Частота дискретизации 100 Гц,
    двухбайтовая беззнаковая кодировка (uint16), 10000 отсчётов на сигнал.

Выход:
    - таблица frequencies.csv со значениями частот для всех сигналов;
    - график graphics/frequency_comparison.png с зависимостью "частота SSA"
      от "частота Фурье" и биссектрисой первой координатной четверти.
"""

import csv
import re
from pathlib import Path
from struct import unpack

import matplotlib
matplotlib.use("Agg")  # сохраняем графики в файл, не открывая окно
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.sparse.linalg import svds

# ---------------------------------------------------------------------------
# Параметры работы
# ---------------------------------------------------------------------------

# частота дискретизации пульсового сигнала, Гц
SAMPLING_RATE = 100.0

# число отсчётов в одном файле (формат распаковки '10000H')
SAMPLES_PER_SIGNAL = 10000

# длина окна (число лагов) при построении траекторной матрицы SSA
SSA_WINDOW = 500

# каталог с исходными сигналами
DATA_DIR = Path("data")

# каталог для графиков и файл с результатами
GRAPHICS_DIR = Path("graphics")
RESULTS_PATH = Path("frequencies.csv")

# шаблон имени файла сигнала: буква d и число, например d0001 или d1023
SIGNAL_NAME_RE = re.compile(r"^d\d+$")


def list_signal_files(data_dir=DATA_DIR):
    """Возвращает отсортированный список файлов с сигналами."""
    files = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and SIGNAL_NAME_RE.match(path.name)
    ]
    files.sort(key=lambda path: path.name)
    return files


def read_signal(path, samples=SAMPLES_PER_SIGNAL):
    """Читает один сигнал из файла.

    Файл хранит samples двухбайтовых беззнаковых чисел (uint16),
    поэтому используется формат распаковки '<samples>H'.
    """
    with open(path, "br") as file:
        raw = file.read()
    return np.array(unpack(f"{samples}H", raw), dtype=np.float64)


def dominant_frequency(signal, sampling_rate=SAMPLING_RATE):
    """Частота и амплитуда главной компоненты Фурье-разложения.

    Постоянная составляющая (нулевая частота) исключается из рассмотрения,
    чтобы не принять средний уровень сигнала за основную гармонику.
    """
    centered = signal - signal.mean()
    spectrum = np.abs(rfft(centered))
    frequencies = rfftfreq(len(centered), d=1.0 / sampling_rate)

    # пропускаем нулевую частоту
    peak_index = int(np.argmax(spectrum[1:])) + 1
    return frequencies[peak_index], spectrum[peak_index]


def build_trajectory_matrix(signal, window):
    """Строит траекторную (главную) матрицу SSA.

    Столбцы матрицы — это скользящие окна длины window по исходному ряду.
    Размер матрицы: (window, N - window + 1).
    """
    n = len(signal)
    columns = n - window + 1
    matrix = np.lib.stride_tricks.as_strided(
        signal,
        shape=(window, columns),
        strides=(signal.strides[0], signal.strides[0]),
    )
    return np.ascontiguousarray(matrix)


def reconstruct_leading_component(signal, window):
    """Восстанавливает первую (главную) компоненту SSA.

    Шаги:
        1. центрирование сигнала;
        2. построение траекторной матрицы;
        3. сингулярное разложение и выбор старшей тройки (u1, s1, v1);
        4. восстановление ряда усреднением по антидиагоналям.

    Усреднение по антидиагоналям rank-1 матрицы s1 * u1 * v1^T сводится
    к свёртке векторов u1 и v1, поделённой на число слагаемых на каждой
    антидиагонали.
    """
    centered = signal - signal.mean()
    n = len(centered)
    matrix = build_trajectory_matrix(centered, window)

    # число слагаемых на каждой антидиагонали (свёртка векторов из единиц)
    counts = np.convolve(np.ones(window), np.ones(matrix.shape[1]))

    try:
        # ищем только старшую тройку — это быстрее полного разложения
        u, s, vt = svds(matrix, k=1, random_state=0, maxiter=2000)
    except Exception:
        # запасной вариант — полное сингулярное разложение
        u_full, s_full, vt_full = np.linalg.svd(matrix, full_matrices=False)
        u, s, vt = u_full[:, :1], s_full[:1], vt_full[:1, :]

    # восстановленный ряд первой компоненты
    component = s[0] * np.convolve(u[:, 0], vt[0, :]) / counts
    return component[:n]


def compute_frequencies(files):
    """Считает частоты Фурье и SSA для всех сигналов.

    Возвращает список кортежей:
        (file, fourier_freq_hz, fourier_amp, ssa_freq_hz, ssa_amp).
    """
    records = []
    total = len(files)

    for position, path in enumerate(files, start=1):
        signal = read_signal(path)

        # частота главной компоненты Фурье-разложения
        f_fourier, a_fourier = dominant_frequency(signal)

        # частота основной гармоники SSA (первая восстановленная компонента)
        leading = reconstruct_leading_component(signal, SSA_WINDOW)
        f_ssa, a_ssa = dominant_frequency(leading)

        records.append((path.name, f_fourier, a_fourier, f_ssa, a_ssa))

        # нечастый вывод прогресса, чтобы видеть ход работы
        if position % 50 == 0 or position == total:
            print(f"Обработано сигналов: {position}/{total}")

    return records


def save_results(records, output_path):
    """Сохраняет таблицу с частотами в CSV-файл."""
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["file", "fourier_freq_hz", "fourier_amp", "ssa_freq_hz", "ssa_amp"]
        )
        writer.writerows(records)
    print(f"Результаты сохранены: {output_path}")


def plot_comparison(records, output_path):
    """Строит график зависимости частоты SSA от частоты Фурье.

    Биссектриса первой координатной четверти служит линией-ориентиром:
    точки вблизи неё означают совпадение оценок двумя методами.
    """
    f_fourier = np.array([row[1] for row in records])
    f_ssa = np.array([row[3] for row in records])

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    # полный диапазон и крупный план области основных частот пульса
    panels = [
        (axes[0], "Полный диапазон", None),
        (axes[1], "Крупный план: 0-3 Гц", 3.0),
    ]

    for ax, title, limit in panels:
        ax.scatter(f_fourier, f_ssa, s=14, alpha=0.6, color="tab:blue", label="сигналы")

        # биссектриса первой координатной четверти
        top = limit if limit is not None else max(f_fourier.max(), f_ssa.max())
        ax.plot([0, top], [0, top], "r--", linewidth=1.5, label="биссектриса")

        ax.set_title(title)
        ax.set_xlabel("Частота главной компоненты Фурье, Гц")
        ax.set_ylabel("Частота основной гармоники SSA, Гц")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if limit is not None:
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)

    figure.suptitle("Сравнение частот Фурье-разложения и SSA")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    print(f"График сохранён: {output_path}")


def main():
    files = list_signal_files()
    if not files:
        raise FileNotFoundError(f"В каталоге {DATA_DIR} не найдено файлов сигналов")

    print(f"Найдено сигналов: {len(files)}")

    records = compute_frequencies(files)

    # сохраняем таблицу с результатами (N значений каждой частоты)
    save_results(records, RESULTS_PATH)

    plot_comparison(records, GRAPHICS_DIR / "frequency_comparison.png")

    # краткая сводка по расхождению двух методов
    difference = np.abs(
        np.array([row[3] for row in records]) - np.array([row[1] for row in records])
    )
    print("\nСводка:")
    print(f"  средняя |f_SSA - f_Фурье| = {np.mean(difference):.4f} Гц")
    print(f"  медианная |f_SSA - f_Фурье| = {np.median(difference):.4f} Гц")
    print(
        "  доля совпадений (|diff| <= 0.05 Гц) = "
        f"{np.mean(difference <= 0.05) * 100:.1f} %"
    )


if __name__ == "__main__":
    main()
