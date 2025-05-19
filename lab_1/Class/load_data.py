from sklearn.datasets import load_iris
from Class.logs import write_exeption

def give_iris():
    try:
        iris = load_iris()
        X = iris.data
        Y = iris.target
        return X,Y, iris
    except Exception as e:
        write_exeption("Class.load_data.give_iris", str(e))
        raise Exception("Can not get data")
