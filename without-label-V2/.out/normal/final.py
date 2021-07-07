import json
import numpy as np

def source_(n_source, n_target, prop_source, prop_target, dist, d, iteration):
    
    filename = f'n_source-{n_source}-n_target-{n_target}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{float(dist)}-d-{d}-iter-{iteration}-normal.json'
    bayes_error, labeled_data, unlabeled_data, mix_classifier, classical_classifier, oracle_unlabeled = np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)) 

    with open(filename, 'r') as fp:
        out = json.load(fp)
        for i in range(100):
            dict_hand = out[str(i)]
            bayes_error[i], labeled_data[i] = dict_hand['test-data']['bayes-error'], dict_hand['labeled-data']['error']
            unlabeled_data[i], mix_classifier[i] = dict_hand['unlabeled-data']['error'], dict_hand['mixture-classifier']['error']
            classical_classifier[i], oracle_unlabeled[i] = dict_hand['classical-classifier']['error'], dict_hand['oracle-classifier']['error']
   
    return np.mean([bayes_error, labeled_data, unlabeled_data, mix_classifier, classical_classifier, oracle_unlabeled], axis  = 1)


n_sources = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
dict_temp = dict()

for n_source in n_sources:
    bayes, labeled, unlabeled, mix, classical, oracle = source_(n_source, 100, 0.5, 0.8, 2, 5, 100)
    dict_temp[str(n_source)] = {'bayes-error': bayes, 'labeled-error': labeled, 'unlabeled-error': unlabeled, 'mix-error': mix, 'classical-error': classical, 'oracle-error': oracle}

with open('source-summary.json','w') as fh:
    json.dump(dict_temp, fh)




n_targets =  [25, 50, 100, 200, 400, 800, 1600, 3200]
dict_temp = dict()



for n_target in n_targets:
    bayes, labeled, unlabeled, mix, classical, oracle = source_(2000, n_target, 0.5, 0.8, 2, 5, 100)
    dict_temp[str(n_target)] = {'bayes-error': bayes, 'labeled-error': labeled, 'unlabeled-error': unlabeled, 'mix-error': mix, 'classical-error': classical, 'oracle-error': oracle}

with open('target-summary.json','w') as fh:
    json.dump(dict_temp, fh)


