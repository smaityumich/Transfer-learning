import numpy as np
from withLabelV3 import *
import pickle
from sklearn.pipeline import Pipeline


xs, xt = np.random.normal(0, 1, (200,1)), np.random.normal(0, 1, (200,1))
b = np.array([1])
ls, lt = xs@b, xt@b
ps, pt = 1/(1+np.exp(-ls)), 1/(1+np.exp(-lt))
ys, yt = np.random.binomial(1, ps), np.random.binomial(1, pt)


class DataHolder():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get(self):
        return self.x, self.y

data = DataHolder(xs, ys)


xsp = pickle.dumps(xs)
ysp = pickle.dumps(ys)


bandwidths = np.linspace(0.1, 2, 2)
param = {'bandwidth': bandwidths}
#grid = GridSearchCV(WithLabelClassifier(), param, cv = 5)
 
cl = WithLabelClassifier() 


#pipe = Pipeline(steps=[('estimator', WithLabelClassifier())])

params_grid = {
                'x_source': [xs],
                'y_source': [ys],
                'bandwidth': bandwidths
                }

l = ParameterGrid(params_grid)
l = list(l)

grid = CVGridSearch(cl, param, cv = 5)
