import numpy as np
from CVGridSearch import *
import sklearn.linear_model

param_dict = {'penalty': ['l1','l2'], 'C': [1, 2]}
method = sklearn.linear_model.LogisticRegression()
grid = CVGridSearch(method, param_dict)
arg = grid.params[0]
method = grid.methods[0]
x = np.random.normal(0, 1, (100,5))
b = np.ones((5,))
l = x@b
y = np.random.binomial(1, 1/(1+np.exp(-l)))
data = x, y

grid.fit(x,y)
