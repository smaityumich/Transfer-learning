# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:47:21 2020

@author: Subha Maity
"""


import sys
sys.path.insert(1, 'D:/GitHub/Tarnsfer-learning/without_label/')
## Change the path to local directory path
## sys.path.insert(1, ${pwd})
import numpy as np
import functools
from densit_ratio import *
from classifier_no_label import *
from data_generator import *


iter_step = 20
n_target = 100
alpha = np.arange(0.5, 2.5, step = 0.25)
source_points = (2**np.array([0,1,2,3,4,5],dtype = int))*100
par = np.meshgrid(source_points, range(iter_step), alpha)
par = np.array(par)
par = par.reshape((3,-1))
par = par.T
dataGenerator = GeneratorClassification()


def _getdata(parameter):
    n_source = parameter[0]
    x_source, y_source = dataGenerator._generate(n_source)
    x_target, y_target = dataGenerator._generate(n_target, prop = 0.8)
    return parameter, (x_source, y_source, x_target, y_target, [dataGenerator._bayesClassifier(_,0.8) for _ in x_target])

def _classify(par_data):
    par, data = par_data
    n_source, _, alpha = par
    h = n_source**(-alpha)
    cl = ClassifierNoLabel(data[0], data[1], data[2])
    cl._targetPropBlackbox(h)
    return par, data[3], [cl._classifyTarget(_, h) for _ in data[2]], data[4]

def _errors(par_labels):
    par, y_true, y_estimate, bayes_label = par_labels
    y_true, y_estimate, bayes_label = np.array(y_true), np.array(y_estimate), np.array(bayes_label)
    bayes_error = np.mean((y_true-bayes_label)**2)
    classifier_error = np.mean((y_estimate-bayes_label)**2)
    print('Done for parameter: '+str(par)+'\n\n')
    return par, classifier_error-bayes_error

data = map(_getdata, par)
classify = map(_classify, data)
error = map(_errors, classify)
error = list(error)
    
    
