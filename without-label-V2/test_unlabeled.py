import numpy as np
from dataGenerator import *
from withoutLabelV3 import *
import sys

data = DataGenerator(4)
n_source, n_target, c = int(float(sys.argv[1])), int(float(sys.argv[2])), float(sys.argv[3])
prop_source = 0.5
prop_target = 0.9
xs, ys, _ = data._getData(n_source, 0.5, 0.4)
xt, yt, _ = data._getData(n_target, 0.9, 0.4)
xtest, ytest, _ = data._getData(200, 0.9, 0.4)

bandwidth =  c*n_source**(-1/6)
cl = WithoutLabelClassifier(workers = 1)
cl.fit(x_source = xs, y_source = ys, x_target = xt, bandwidth = bandwidth)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Error BBLS {error}\n')

bandwidth = c*n_source**(-1/6)
cl = KDEClassifier(bandwidth)
w = np.array([(1-prop_target)/(1-prop_source), prop_target/prop_source])
cl.fit(X = xs, y = ys, weights = w)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Error Oracle {error}\n')


