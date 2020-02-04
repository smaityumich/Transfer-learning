import numpy as np
from withoutLabelV2 import *
import sys
import os
from fractions import Fraction


np.random.seed(100)
m, n, d, prop, distance = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
m = float(m)
n = float(n)
d = float(d)
prop = float(prop)
distance = float(distance)
m, n, d = int(m) , int(n), int(d)
fname = f'n-source:{m} n-target:{n} dimension:{d} prop-of-success-target:{prop} dist-between-means: {distance}'
#os.system(f'touch experiment2.out')

data_generate = DataGenerator(d = d)
x_source, y_source, _ = data_generate._getData(m, 0.5, distance)
x_target, y_target, bayes_target = data_generate._getData(n, prop, distance)

classifier_noLabel = WithoutLabelV2(x_source = x_source, y_source = y_source, x_target = x_target)
predicted_labels = classifier_noLabel._classify()
error = np.mean((y_target-predicted_labels)**2)
bayes_error = np.mean((bayes_target-y_target)**2)
w = classifier_noLabel.w
w_true = [(1-prop)/0.5, prop/0.5]
w_true = np.array(w_true)
w_error = np.sum((w - w_true)**2)
bandwidth = classifier_noLabel.bandwidth

with open('experiment2.out','a') as fh:
    fh.writelines(f'parameter: \n {fname}\n')
    fh.writelines(f'Prediction error: {error}\n')
    fh.writelines(f'Bayes error: {bayes_error}\n')
    fh.writelines(f'w: {str(w)}\n w_error: {w_error}\n\n\n')
    
os.system(f'echo Prediction error {error}')
os.system(f'echo Bayes error: {bayes_error}')
    




