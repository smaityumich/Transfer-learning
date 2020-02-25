from experiments import *
import sys
import json

n_source, n_target, n_test, prop_source, prop_target, dist, d, iteration = int(float(sys.argv[1])), int(float(sys.argv[2])), int(float(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(float(sys.argv[7])), int(float(sys.argv[8]))
outputs = dict()
outputs['parameters'] = {'n_source': n_source, 'n_target': n_target, 'n_test': n_test, 'prop_source': prop_source, 'prop_target': prop_target, 'dist': dist, 'dimension': d, 'iter': iteration}

for i in range(iteration):
    e = Experiments()
    e._getData(n_source, n_target, n_test, prop_source, prop_target, dist, d)
    e._QLabledClassifier()
    e._QUnlabeledClassifier()
    e._MixtureClassifier()
    e._ClassicalClassifier()
    e._OracleClassifierNoTargetLabel()
    outputs[i] = e.output

filename = f'.out/n_source-{n_source}-n_target-{n_target}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}-beta.json'

with open(filename, 'w') as fp:
    json.dump(outputs, fp)


