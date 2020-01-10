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


iter_step = 100
n_target = 100
source_points = (2**np.array([0,1,2,3,4,5],dtype = int))*100
par = np.meshgrid(source_points, range(iter_step))
par = np.array(par)
par = par.reshape((2,-1))
par = par.T
dataGenerator = GeneratorClassification()


def _getdata(parameter):
    n_source = parameter[0]
    x_source, y_source = dataGenerator._generate(n_source)
    x_target, y_target = dataGenerator._generate(n_target, prop = 0.8)
    return parameter, (x_source, y_source, x_target, y_target)

def _classify(par_data):
    par, data = par_data
    cl = ClassifierNoLabel(data[0], data[1], data[2])
    
