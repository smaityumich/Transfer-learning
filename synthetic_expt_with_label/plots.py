# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:52:51 2019

@author: Subha Maity
"""
import numpy as np
import pickle 
f = open("C:/Users/Subha Maity/OneDrive/Documents/GitHub/transferlearning/synthetic_expt/gen_error_exp.pckl", 'rb')
pickle.load(f)
f.close()


n_source = [50*2**x for x in list(range(10))]

gen_nn = list(map(lambda x: x[1][1], gen_error))
gen_nn = np.array(gen_nn)
gen_nn = gen_nn.reshape((10,-1))
nn_median = np.median(gen_nn, axis = 1)
nn_upper = np.percentile(gen_nn, 0.75, axis = 1)
nn_lower = np.percentile(gen_nn, 0.25, axis = 1)









gen_ts = list(map(lambda x: x[1][0], gen_error))
gen_ts = np.array(gen_ts)
gen_ts = gen_ts.reshape((10,-1))
ts_median = np.median(gen_ts, axis = 1)
ts_upper = np.percentile(gen_ts, 0.75, axis = 1)
ts_lower = np.percentile(gen_ts, 0.25, axis = 1)


import matplotlib.pyplot as plt

plt.errorbar(n_source,nn_median, yerr=(nn_upper-nn_lower)/2)
plt.plot(n_source,gen_nn,'r--')


from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(n_source),np.log(gen_ts))
