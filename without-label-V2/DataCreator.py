from dataGenerator import *
import sys
import json


if len(sys.argv) != 9:
    raise TypeError('Wrong input number')
    sys.exit(1)


n_source, n_target, n_test, prop_source, prop_target, dist, d, iteration = int(float(sys.argv[1])), int(float(sys.argv[2])), int(float(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(float(sys.argv[7])), int(float(sys.argv[8]))





datagenerator = DataGenerator(d = d)
data = dict()

data['source'] = dict()
xs, ys, _ = datagenerator._getData(n = n_source, prop = prop_source, distance = dist)
data['source']['x'], data['source']['y'] = xs.tolist(), ys.tolist()

data['target'] = dict()
xt, yt, _ = datagenerator._getData(n = n_target, prop = prop_target, distance = dist)
data['target']['x'], data['target']['y'] = xt.tolist(), yt.tolist()


data['test'] = dict()
xt, yt, _ = datagenerator._getData(n = n_test, prop = prop_target, distance = dist)
data['test']['x'], data['test']['y'] = xt.tolist(), yt.tolist()








filename = f'.data/n_source-{n_source}-n_target-{n_target}-n_test-{n_test}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}-normal-data.json'

with open(filename, 'w') as fp:
    json.dump(data, fp)


