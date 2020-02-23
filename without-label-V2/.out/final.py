import json
import numpy as np

def source_(n_source, n_target, prop_source, prop_target, dist, d, iteration):
    
    filename = f'n_source-{n_source}-n_target-{n_target}-prop_source-{prop_source}-prop_target-{prop_target}-dist-{dist}-d-{d}-iter-{iteration}.json'
    bayes_error, labeled_data, unlabeled_data, mix_classifier, classical_classifier, oracle_unlabeled = np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)), np.zeros((iteration,)) 

    with open(filename, 'r') as fp:
        out = json.load(fp)
        for i in range(100):
            dict_hand = out[str(i)]
            bayes_error[i], labeled_data[i] = dict_hand['test-data']['bayes-error'], dict_hand['labeled-data']['error']
            unlabeled_data[i], mix_classifier[i] = dict_hand['unlabeled-data']['error'], dict_hand['mixture-classifier']['error']
            classical_classifier[i], oracle_unlabeled[i] = dict_hand['classical-classifier']['error'], dict_hand['oracle-classifier']['error']
   
    return np.array([bayes_error, labeled_data, unlabeled_data, mix_classifier, classical_classifier, oracle_unlabeled])
