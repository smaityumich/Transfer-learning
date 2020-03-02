import json

list_json = []

def f(n_source, n_target, n_test,  prop_source, prop_target, dist, d, iteration, experiment):
    filename = f'.result/n_source-{n_source}-n_target-{n_target}-n_test-{n_test}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}-experiment-{experiment}-normal-result.json'

    with open(filename, 'w') as fp:
        d = json.load(fp)
        list_json.append(d)




