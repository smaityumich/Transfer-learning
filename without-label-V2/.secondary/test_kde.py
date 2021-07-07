from dataGenerator import *
from kdeClassifier import *

data = DataGenerator(d = 5)
x, y, _  = data._getData(200, 0.8, 2)
cl = KDEClassifierOptimalParameter()
cl.fit(x, y)
yp = cl.predict(x)
