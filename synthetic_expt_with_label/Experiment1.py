# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:51:16 2019

@author: Subha Maity
"""
import sys
import numpy as np
import functools
sys.path.insert(0, "C:/Users/Subha Maity/OneDrive/Documents/GitHub/transferlearning/synthetic_expt/")
import Simulate as sim
import ClassifierTS as ts

num_points = 50   
            
def get_data(key):
    s = sim.SimulateClassification(3)
    x_P, y_P = s.simulateP_(int(key[1]))
    x_Q, y_Q = s.simulateQ_(20)
    x_test, bayes = s.bayes_rule_(num_points)
    return [(key, [x_P, y_P, x_Q, y_Q, x_test[i], bayes[i]]) for i in range(len(x_test))]


def classify(y):
    key, data = y
    print(str(key))
    s = sim.SimulateClassification(3)
    cl = ts.AdaClassifier(data[0],data[1],data[2],data[3], kernel="cauchy")
    predict = cl.predict_one(data[4])
    predict_knn = cl.knn_(data[2],data[3],data[4])
    prop = np.absolute(s.prop_(data[4]) - 0.5)
    return key, [np.absolute(data[5]-predict)*prop, np.absolute(data[5]-predict_knn)*prop]

def classify_list(y_list):
    return [classify(y) for y in y_list]



n_source = [50*2**x for x in list(range(10))]
ITER = range(40)
par = np.meshgrid(ITER, n_source)
par = np.array(par)
par = par.reshape((2,-1))
par = par.T
par = [tuple(y) for y in par]







def vec_sum(x,y):
    return [(x[i]+y[i]) for i in range(len(x))]

data = map(get_data, par)
error = map(classify_list, data)
error_list = list(error)

def generalized_error(y):
    key = y[0][0]
    value = [z[1] for z in y]
    gen_error = functools.reduce(vec_sum, value, [0,0])
    gen_error = [y/num_points for y in gen_error]
    
    return key, gen_error


gen_error = map(generalized_error, error_list)
gen_error = list(gen_error)
gen_error = list(map(lambda x: (x[0][:2],x[1]),gen_error))
gen_error



import pickle 
f = open("C:/Users/Subha Maity/OneDrive/Documents/GitHub/transferlearning/synthetic_expt/gen_error_cauchy.pckl", 'wb')
pickle.dump(gen_error, f)
f.close()
