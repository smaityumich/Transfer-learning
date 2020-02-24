import numpy as np
from withLabelV3 import *
import multiprocessing as mp
from dataGenerator import *


def unit_iter_labeled(par = (500, 200, 200, 0.5, 0.8, 1, 4)):
    
    # Parameters 
    n_source, n_target, n_test, prop_source, prop_target, dist, d = par
    n_source, n_target, n_test, d = int(n_source), int(n_target), int(n_test), int(d)

    # Data
    data = DataGenerator(d = d)
    x_source, y_source, _ = data._getData(n_source, prop_source, dist)
    x_target, y_target, _ = data._getData(n_target, prop_target, dist)
    x_test, y_test, bayes_test = data._getData(n_test, prop_target, dist)

    # Classifier 
    cl = WithLabelOptimalClassifier(nodes = mp.cpu_count())
    cl.fit(x_source, y_source, x_target, y_target)
    y_predict = cl.predict(x_test)

    return np.mean((y_predict - y_test)**2) - np.mean((bayes_test-y_test)**2)



def unit_expt_labeled(n_source = 500, n_target = 50, n_test = 100, prop_source = 0.5, prop_target = 0.8, dist = 1, d = 4, iteration = 10):
    
    par = n_source, n_target, n_test,  prop_source, prop_target, dist, d
    pars = [par for _ in range(iteration)]


    out = map(unit_iter_labeled, pars)

    out = list(out)
    out = np.array(out)
    return np.mean(out, axis = 0)


n_sources = [25, 50, 100, 200, 400, 800, 1600, 3200]

print('n_sources: ')
print(n_sources)
print('\n\n\n')

for n_source in n_sources:

    out = unit_expt_labeled(n_source = n_source)
    print(f'n_source: {n_source}\n')
    print(out)
    print('\n\n')


