import numpy as np

def epanechnikov_kernel(dist):
		if abs(dist) <= 1:
				return (3 / 4) * (1 - (dist * dist))
		else:
				return 0

def evc_distance(x, y):
		return np.sqrt(np.sum((x - y) ** 2))