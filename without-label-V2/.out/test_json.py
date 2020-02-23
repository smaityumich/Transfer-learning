import json
import numpy as np

filename = f'n_source-{100}-n_target-{200}-prop_source-{0.5}-prop_target-{0.8}-dist-{0.8}-d-{5}-iter-{100}.json'
fp = open(filename, 'r')
out = json.load(fp)

bayes_error, labeled_data, unlabeled_data, mix_classifier, classical_classifier, oracle_unlabeled = np.zeros((100,)), np.zeros((100,)), np.zeros((100,)), np.zeros((100,)), np.zeros((100,)), np.zeros((100,)) 
for i in range(100):
    dict_hand = out[str(i)]
    bayes_error[i], labeled_data[i] = dict_hand['test-data']['bayes-error'], dict_hand['labeled-data']['error']
    unlabeled_data[i], mix_classifier[i] = dict_hand['unlabeled-data']['error'], dict_hand['mixture-classifier']['error']
    classical_classifier[i], oracle_unlabeled[i] = dict_hand['classical-classifier']['error'], dict_hand['oracle-classifier']['error']
