from mixtureClassifier import *
import numpy as np

xs, xt = np.random.normal(0, 1, (200,1)), np.random.normal(0, 1, (200,1))
b = np.array([1])
ls, lt = xs@b, xt@b
ps, pt = 1/(1+np.exp(-ls)), 1/(1+np.exp(-lt))
ys, yt = np.random.binomial(1, ps), np.random.binomial(1, pt)


cl = OptimalMixtureClassifier()
cl.fit(xs, ys, xt, yt)

