from kdeClassifier import *
from withLabelV3 import *
import numpy as np
from dataGenerator import *
from mixtureClassifier import *
import sys

n_source, n_target, c = int(float(sys.argv[1])), int(float(sys.argv[2])), float(sys.argv[3])

n_test = 100

data = DataGenerator(4)
xs, ys, _ = data._getData(n_source)
xt, yt, _ = data._getData(n_target, prop = 0.9)
xtest, ytest, _ = data._getData(prop = 0.9)


bandwidth = 2*c*(n_source+n_target)**(-1/6)
cl = WithLabelOptimalClassifier(nodes = 1)
cl.fit(x_source = xs, y_source = ys, x_target = xt, y_target = yt, bandwidth = bandwidth)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Both c:{c} error:{error}')


bandwidth = c*(n_target)**(-1/6)
cl = KDEClassifier(bandwidth = bandwidth)
cl.fit(X = xt, y = yt)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Classical c:{c} error:{error}')

cl = OptimalMixtureClassifier(nodes = 1)
bandwidth_source = c*(n_source)**(-1/6)
bandwidth_target = c*(n_target)**(-1/6)
cl.fit(x_source = xs, y_source = ys, x_target = xt, y_target = yt, bandwidth_source=bandwidth_source, bandwidth_target=bandwidth_target)
y_pred = cl.predict(xtest)
error = np.mean((y_pred - ytest)**2)
print(f'Mixture c:{c} error:{error}')



