import numpy as np
import matplotlib.pyplot as plt
from Class.Bayas import fit_gaus, fit_laplace, pred
from Class.get_data import load_dataset
from Class.function import gaussian, laplace
from sklearn.metrics import f1_score

def main():
    X_train, y_train, X_test, y_test = load_dataset()
    M_gaus, D_gaus, prior_gaus, y_gaus = fit_gaus(X_train, y_train)
    pred_gaus = pred(X_test, M_gaus, D_gaus, prior_gaus, y_gaus, gaussian)

    M_lapl, B_lapl, prior_lapl, y_lapl = fit_laplace(X_train, y_train)
    pred_lapl = pred(X_test, M_lapl, B_lapl,prior_lapl, y_lapl, laplace)
    
    
    
    Q_gaus = f1_score(y_test, pred_gaus, average='macro')
    Q_lapl = f1_score(y_test, pred_lapl, average='macro')

    plot_x = X_train[y_train == 0][:, 0]
    mu_gaus = M_gaus[0][0]
    var_gaus = D_gaus[0][0]
    mu_lapl = M_lapl[0][0]
    b_lapl = B_lapl[0][0]

    x_vals = np.linspace(min(plot_x) - 1, max(plot_x) + 1, 100)
    gaus = gaussian(x_vals, mu_gaus, var_gaus)
    lapl = laplace(x_vals, mu_lapl, b_lapl)

    plt.figure(figsize=(10, 6))
    plt.hist(plot_x, bins=10, density=True, alpha=0.6)
    plt.plot(x_vals, gaus, 'r-')
    plt.plot(x_vals, lapl, 'g--')

    plt.title(f'Класс 0, признак 0')
    plt.grid(True)
    plt.show()

    print(f"Качество нормальное распр: {Q_gaus}")
    print(f"Качество лапласовское распр: {Q_lapl}")

main()