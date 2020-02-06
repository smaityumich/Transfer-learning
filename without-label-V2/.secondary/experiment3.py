import numpy as np
from withoutLabelV2 import *
import sys
import os
from fractions import Fraction


## Mkaing a hidden output directory
outdir = os.getcwd() + '/.out'
if not os.path.exists(outdir):
    os.system(f'mkdir {outdir}')


## Getting classifier for the 



def job(par):

    ## Setting seed and parameter
    #np.random.seed(100)
    m, n, n_test, d, prop, distance, index  = par
    #m = float(m)
    #n, n_test = float(n), float(n_test)
    #d = float(d)
    #prop = float(prop)
    #distance = float(distance)
    #m, n, n_test, d = int(m) , int(n), int(n_test), int(d)
    fname = f'n-source:{m} n-target:{n} dimension:{d} prop-of-success-target:{prop} dist-between-means: {distance} index {index}'







    ##Generate data
    data_generate = DataGenerator(d = d)
    x_source, y_source, _ = data_generate._getData(m, 0.5, distance)
    x_target_train, _, _ = data_generate._getData(n, prop, distance)
    x_target_test, y_target_test, bayes_target_test = data_generate._getData(n_test, prop, distance)


    ##Buliding classifier 
    classifier_noLabel = WithoutLabelV2(x_source = x_source, y_source = y_source, x_target = x_target_train)
    predicted_labels = classifier_noLabel._classify(x_target_test)
    error = np.mean((y_target_test-predicted_labels)**2)
    bayes_error = np.mean((bayes_target_test-y_target_test)**2)
    w = classifier_noLabel.w
    w_true = [(1-prop)/0.5, prop/0.5]
    w_true = np.array(w_true)
    w_error = np.sum((w - w_true)**2)
    bandwidth = classifier_noLabel.bandwidth
    prop_predicted = classifier_noLabel.prop_target
    prop_error = np.abs(prop_predicted-prop)

    with open('./out/experiment.out','a') as fh:
        fh.writelines(f'parameter:\n{fname}\n')
        fh.writelines(f'Prediction error: {error}\n')
        fh.writelines(f'Bayes error: {bayes_error}\n')
        fh.writelines(f'w: {str(w)}\nw_error: {w_error}\n')
        fh.writelines(f'Bandwidth chosen {bandwidth}\n')
        fh.writelines(f'Target proportion is estimated as {prop_predicted} with error {prop_error}\n\n\n')
    os.system(f'echo Excess risk: {error-bayes_error}')
    os.system(f'echo w_error: {w_error}')
    os.system(f'echo Target proportion estimation error {prop_error}')    




