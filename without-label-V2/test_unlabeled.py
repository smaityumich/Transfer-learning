import numpy as np
from dataGenerator import *
from withoutLabelV3 import *

data = DataGenerator(4)

xs, ys, _ = data._getData(400, 0.5, 0.4)
xt, yt, _ = data._getData(100, 0.8, 0.4)
xtest, ytest, _ = data._getData(200, 0.8, 0.4)

bandwidth =  0.6*400**(-1/6)
cl = WithoutLabelClassifier(workers = 1)
cl.fit(x_source = xs, y_source = ys, x_target = xt, bandwidth = bandwidth)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Error {error}\n')


