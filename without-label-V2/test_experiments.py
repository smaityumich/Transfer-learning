from experiments import *
import cProfile


def single_iteration(n_source = 100, n_target = 100, n_test = 20, prop_source = 0.5, prop_target = 0.8, dist = 0.5, d = 4):
    e = Experiments()
    e._getData(n_source, n_target, n_test, prop_source, prop_target, dist, d)
    e._QLabledClassifier()
    e._QUnlabeledClassifier()
    e._MixtureClassifier()
    e._ClassicalClassifier()
    e._OracleClassifierNoTargetLabel()
    return e

e = single_iteration()
