# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:22:36 2020

@author: Subha Maity
"""
import sys
sys.path.insert(1, 'D:/GitHub/Tarnsfer-learning/without_label/')
## Change the path to local directory path
## sys.path.insert(1, ${pwd})
import numpy as np
import functools
from classifier_no_label import *
from data_generator import *
import seaborn as sns
import pandas as pd
import sklearn.metrics
import scipy



iter_step = 20
n_target = 200
alpha = np.arange(0.5, 2.5, step = 0.5)
source_points = (2**np.array([0, 2, 4, 6],dtype = int))*100
par = np.meshgrid(source_points, range(iter_step), alpha)
par = np.array(par)
par = par.reshape((3,-1))
par = par.T
dataGenerator = GeneratorClassification()


par = par[0]



def _getdata(parameter):
    n_source = parameter[0]
    x_source, y_source = dataGenerator._generate(n_source)
    x_target, y_target = dataGenerator._generate(n_target, prop = 0.8)
    return parameter, (x_source, y_source, x_target, y_target, np.array([dataGenerator._bayesClassifier(_,0.8) for _ in x_target]))

def _classify(par_data):
    par, data = par_data
    n_source, _, alpha = par
    bandwidth = n_source**(-alpha)
    cl = ClassifierNoLabel(bandwidth= bandwidth)
    cl._data(data[0],data[1], data[2])
    _, target_prop_estimate = cl._targetPropBlackbox()
    return par, data[3], cl._classifyTarget(data[2]), data[4], target_prop_estimate[1]

def _errors(par_labels):
    par, y_true, y_estimate, bayes_label, target_prop_estimate = par_labels
    y_true, y_estimate, bayes_label = np.array(y_true), np.array(y_estimate), np.array(bayes_label)
    bayes_error = np.mean((y_true-bayes_label)**2)
    classifier_error = np.mean((y_estimate-bayes_label)**2)
    print('Done for parameter: '+str(par)+'\n\n')
    return par, classifier_error-bayes_error, np.absolute(target_prop_estimate - 0.8)




data = _getdata(par)
param, data = data
x_s, y_s, x_t, y_t, b_t = data
n_source, _, alpha = param
bandwidth = n_source**(-alpha)
cl = ClassifierNoLabel(bandwidth = bandwidth)
cl._data(x_s, y_s, x_t)

w, pi = cl._targetPropBlackbox()

pi_y = cl._targetPropEstimateYK()


predict_source = cl._classifySource(cl.x_source)  #predicted source lables
predict_target = cl._classifySource(cl.x_target)  #predicted target labels
confusion_mx = sklearn.metrics.confusion_matrix(cl.y_source, predict_source)/len(predict_source)
y_hat_target = np.mean(predict_target)
mu_hat = [1-y_hat_target, y_hat_target]
w_hat = np.matmul(scipy.linalg.inv(confusion_mx), mu_hat)



