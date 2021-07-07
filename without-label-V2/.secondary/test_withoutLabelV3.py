from dataGenerator import *
from withoutLabelV3 import *


datagenerator = DataGenerator(d = 5)
xs, ys, _ = datagenerator._getData(200, 0.5, 2)
xt, yt, _ = datagenerator._getData(200, 0.8, 2)

cl = WithoutLabelClassifier(workers = 2)
cl.fit(xs, ys, xt)
cl.predict(xt)
