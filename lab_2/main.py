import numpy as np
import Class.Paint_data as painter
from ucimlrepo import fetch_ucirepo
import Class.Claster as cls


iris = fetch_ucirepo(id=53)
X = iris.data.features.values
k = 3
#k = int(input("Count clusters: "))
labels, centers, qual = cls.KMeans(X, k)

print(f"Quality: {qual}")
for i in range(len(centers)):
    print(f"Centre {i}: {centers[i]}")

painter.plot_Kmens(X,labels,centers,k)