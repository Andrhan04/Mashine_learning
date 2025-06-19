import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(C_arr, log_acc, svc_acc):
    # строит график зависимости accuracy от параметра регуляризации C
    plt.plot(C_arr, log_acc, label = 'LogReg', marker = 'o')
    plt.plot(C_arr, svc_acc, label = 'LinSVC', marker = 'x')

    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Точность при C')
    plt.legend()
    plt.savefig("images\\accuracy.png")
    plt.show()


def plot_coeffs(C_arr, coefs, title):
    # График показывает, как значения коэффициентов модели изменяются при различной силе регуляризации

    coefs = np.array(coefs)
    for i in range(coefs.shape[1]):
        plt.semilogx(C_arr, coefs[:, i], label = f'Feature {i}')
    
    plt.title(f'{title} Значение на С')
    plt.xlabel('C')
    plt.ylabel('Коэффициенты (веса признаков)')
    plt.tight_layout()   # вычисляет оптимальные значения отступов между элементами
    plt.savefig("images\\Coef.png")
    plt.show()
