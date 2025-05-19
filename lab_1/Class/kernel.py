import numpy as np
from Class.logs import  write_exeption

def epanechnikov_kernel(dist):
    try:
        if abs(dist) <= 1:
            return (3 / 4) * (1 - (dist * dist))
        else:
            return 0
    except Exception as e:
        write_exeption("Class.kernel.epanechnikov_kernel", str(e))
        raise Exception("Can not get kernel")

def evc_distance(x, y):
    try:
        return np.sqrt(np.sum((x - y) ** 2))
    except Exception as e:
        write_exeption("Class.kernel.evc_distance", str(e))
        raise Exception("Can not get distance")