import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Class.logs import write_exeption

def show_dataset(data : np.ndarray, answer  : np.ndarray):
    try:
        for i in range(len(data)):
            a,b,c,d = data[i]
            print( f"{a} {b} {c} {d} |{answer[i]}")
    except Exception as e:
        write_exeption("Class.Paint_data.show_dataset", str(e))
        raise Exception("Can not show data")

def plot_pairplot(dataset : np.ndarray, X : np.ndarray, Y : np.ndarray):
    try:
        df = pd.DataFrame(X, columns = dataset.feature_names)
        df["target"] = [dataset.target_names[i] for i in Y]
        print(df)
        sns.pairplot(df, hue = "target")
        plt.savefig("images\\Classes.png")
        plt.show()
        plt.close()
    except Exception as e:
        write_exeption("Class.Paint_data.plot_loo_error", str(e))
        raise Exception("Can not show data")

def plot_loo_error(k_vals : list):
    try:
        plt.bar(range(1, len(k_vals) + 1), k_vals)
        plt.xlabel("k")
        plt.ylabel("LOO Accuracy")
        plt.title("LOO Error vs k")
        for i, val in enumerate(k_vals):   # получаем и индекс и значение 
                plt.text(i + 1, val, str(val), ha = "center", va = "bottom", fontsize = 8)
        plt.savefig("images\\LOO_Error_vs_k.png")
        plt.show()
        plt.close()
    except Exception as e:
        write_exeption("Class.Paint_data.plot_loo_error", str(e))
        raise Exception("Can not show data")

def plot_Kmens(X, labels, centers,k):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    for i in range(k):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], color=colors[i % 6])
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='X', linewidths = 1, s=200) 
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.savefig(f"images\\Classters_{k}.png")
    plt.show()