from dataGenerator import *
from withLabelV3 import *

data = DataGenerator(5)
xs, ys, _ = data._getData(200, 0.5, 2)
xt, yt, _ = data._getData(100, 0.8, 2)
cl = WithLabelOptimalClassifier()
cl.fit(xs, ys, xt, yt)
