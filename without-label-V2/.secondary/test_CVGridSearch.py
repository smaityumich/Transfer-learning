import numpy as np
from CVGridSearch import *
from withLabelV3 import *


param_dict = {'bandwidth': np.linspace(0.1, 2, 20)}
method = WithLabelClassifier()
grid = CVGridSearch(method, param_dict)
xs, xt = np.random.normal(0, 1, (200, 3)), np.random.normal(0, 1, (200, 3))
b = np.array([1,1,1])
ps, pt = 1/(1+np.exp(-xs@b)), 1/(1+np.exp(-xt@b))
ys, yt = np.random.binomial(1, ps), np.random.binomial(1, pt)
grid.fit(xs, ys, xt, yt)


cl = WithLabelOptimalClassifier()
cl.fit(xs, ys, xt, yt)
