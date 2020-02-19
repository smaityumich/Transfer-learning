import numpy as np
from withLabelV2 import *

xs, xt = np.random.normal(0, 1, (200, 3)), np.random.normal(0, 1, (200, 3))
bs, bt = xs@np.array([1,1,1]), xt@np.array([1,1,1])
ys, yt = np.random.binomial(1, 1/(1+np.exp(-bs))), np.random.binomial(1, 1/(1+np.exp(-bt)))
sourceData = xs, ys
xtest = np.random.normal(0, 1, (50, 3))
param = {'bandwidth': np.linspace(0.1, 2, 20)}


#grid = GridSearchCV(WithLabelClassifier(), param, cv = 5)

