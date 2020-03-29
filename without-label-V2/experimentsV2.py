from kdeClassifier import *
from withLabelV3 import *
from withoutLabelV3 import *
import numpy as np
from dataGenerator import *
from mixtureClassifier import *
import multiprocessing as mp
import sys
import json

if len(sys.argv) != 10:
    raise TypeError('Wrong input number')
    sys.exit(1)


n_source, n_target, n_test, prop_source, prop_target, dist, d, iteration, experiment = int(float(sys.argv[1])), int(float(sys.argv[2])), int(float(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(float(sys.argv[7])), int(float(sys.argv[8])), str(sys.argv[9])


filename = f'.data/n_source-{n_source}-n_target-{n_target}-n_test-{n_test}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}-normal-data.json'

with open(filename, 'r') as fp:
    data = json.load(fp)

dh = data['source']
xs, ys = np.array(dh['x']), np.array(dh['y'])

dh = data['target']
xt, yt = np.array(dh['x']), np.array(dh['y'])

dh = data['test']
xtest, ytest = np.array(dh['x']), np.array(dh['y'])


result = dict()

if experiment == 'QLabeled':
    bandwidth = (n_source+n_target)**(-1/6)
    cl = WithLabelOptimalClassifier(nodes = 1)
    cl.fit(x_source = xs, y_source = ys, x_target = xt, y_target = yt, bandwidth = bandwidth)
    result['bandwidth'] = cl.bandwidth
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)


elif experiment == 'QUnlabeled':
    cl = WithoutLabelClassifier(workers = 1)
    cl.fit(x_source = xs, y_source = ys, x_target = xt)
    result['bandwidth'] = cl.bandwidth
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)

elif experiment == 'Mixture':
    cl = OptimalMixtureClassifier(nodes = 1)
    cl.fit(x_source = xs, y_source = ys, x_target = xt, y_target = yt)
    result['bandwidth'] = cl.mixture
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)


elif experiment == 'Mixture_bandwidth':
    cl = OptimalMixtureClassifier(nodes = 1)
    bandwidth_source = 0.5*(n_source)**(-1/6)
    bandwidth_target = 0.5*(n_target)**(-1/6)
    cl.fit(x_source = xs, y_source = ys, x_target = xt, y_target = yt, bandwidth_source=bandwidth_source, bandwidth_target=bandwidth_target)
    result['bandwidth'] = cl.mixture
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)



elif experiment == 'Classical':
    cl = KDEClassifierOptimalParameter(workers = 1)
    cl.fit(x = xt, y = yt)
    result['bandwidth'] = cl.bandwidth
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)

elif experiment == 'Classical_bandwidth':
    bandwidth = 0.5*(n_target)**(-1/6)
    cl = KDEClassifier(bandwidth = bandwidth)
    cl.fit(X = xt, y = yt)
    result['bandwidth'] = bandwidth
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)


elif experiment == 'Oracle':
    cl = KDEClassifierOptimalParameter(workers = 1)
    cl.fit(x = xs, y = ys)
    result['bandwidth'] = cl.bandwidth
    cl = KDEClassifier(result['bandwidth'])
    w = np.array([(1-prop_target)/(1-prop_source), prop_target/prop_source])
    cl.fit(X = xs, y = ys, weights = w)
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)

elif experiment == 'Oracle_bandwidth':
    bandwidth = 0.6*n_source**(-1/6)
    cl = KDEClassifier(bandwidth)
    w = np.array([(1-prop_target)/(1-prop_source), prop_target/prop_source])
    cl.fit(X = xs, y = ys, weights = w)
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)

elif experiment == 'QUnlabeled_bandwidth':
    bandwidth =  0.6*n_source**(-1/6)
    cl = WithoutLabelClassifier(workers = 1)
    cl.fit(x_source = xs, y_source = ys, x_target = xt, bandwidth = bandwidth)
    result['bandwidth'] = cl.bandwidth
    y_pred = cl.predict(xtest)
    result['error'] = np.mean((y_pred - ytest)**2)


else:
    raise TypeError('Wrong experiment')


result['n-source'], result['n-target'], result['n-test'], result['prop-source'], result['prop-target'], result['dist'], result['d'], result['iter'], result['expt'] =  n_source, n_target, n_test, prop_source, prop_target, dist, d, iteration, experiment
 


filename = f'.result/n_source-{n_source}-n_target-{n_target}-n_test-{n_test}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}-expt-{experiment}-normal-data.json'

with open(filename, 'w') as fp:
    json.dump(result, fp)
  

