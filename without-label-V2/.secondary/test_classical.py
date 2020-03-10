from dataGenerator import *
from kdeClassifier import *
import numpy as np

def f(n = 50, prop = 0.8, dist = 0.8):
    data = DataGenerator(5)
    _, yt, bayes = data._getData(500000, prop, dist)
    return np.mean((bayes-yt)**2)


print(f())

