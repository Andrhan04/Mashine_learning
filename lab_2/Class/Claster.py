import numpy as np

def Dist(X, test_point):
    return np.sqrt(np.sum((X - test_point) ** 2, axis=1))


def InitCenters(k, X):
    return X[np.random.choice(len(X), k, replace=False)]


def Expect(X, centers):
    labels = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        dist = Dist(centers, X[i])
        labels[i] = np.argmin(dist)
    return labels


def UpdateCenters(labels : np.ndarray, X : np.ndarray, k : int):
    new_centers = np.zeros((k, X.shape[1]), dtype = float)
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centers[i] = np.mean(cluster_points, axis = 0)
    return new_centers


def Quality(labels : np.ndarray, centers : np.ndarray, X : np.ndarray):
    total = 0
    for i in range(len(centers)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            dist = Dist(cluster_points, centers[i])
            total += np.sum(dist ** 2)
    return total


def KMeans(X, k):
    best_qual = float('inf')
    best_labels = None
    best_centers = None

    for _ in range(20):
        centers = InitCenters(k, X)
        prev_qual = float('inf')
        while True:
            labels = Expect(X, centers)
            centers = UpdateCenters(labels, X, k)
            cur_qual = Quality(labels, centers, X)
            if cur_qual >= prev_qual or cur_qual == 0:
                break
            prev_qual = cur_qual
        if cur_qual < best_qual:
            best_qual = cur_qual
            best_labels = labels
            best_centers = centers
    return best_labels, best_centers, best_qual
