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
from classifier_no_label import *
from data_generator import *
import seaborn as sns
import pandas as pd


iter_step = 20
n_target = 200
alpha = np.arange(0.5, 2.5, step = 0.5)
source_points = (2**np.array([0, 2, 4, 6],dtype = int))*100
par = np.meshgrid(source_points, range(iter_step), alpha)
par = np.array(par)
par = par.reshape((3,-1))
par = par.T
dataGenerator = GeneratorClassification()


def _getdata(parameter):
    n_source = parameter[0]
    x_source, y_source = dataGenerator._generate(n_source)
    x_target, y_target = dataGenerator._generate(n_target, prop = 0.7)
    return parameter, (x_source, y_source, x_target, y_target, [dataGenerator._bayesClassifier(_,0.8) for _ in x_target])

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

data = map(_getdata, par)
classify = map(_classify, data)
error = map(_errors, classify)
error = list(error)
error = np.array(error)
np.save('D:/GitHub/Tarnsfer-learning/without_label/result.npy', error)


sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")

n_source, alpha, gen_error, prop_error = np.zeros(len(error)), np.zeros(len(error)), np.zeros(len(error)), np.zeros(len(error))

for i in range(len(error)):
    par, gen_error[i], prop_error[i] = error[i]
    n_source[i], _, alpha[i] = par
    
    
errors = pd.DataFrame(np.array([n_source, alpha, gen_error, prop_error]).T, columns = ['n_source', '$\\alpha$', 'gen_error', 'prop_error'])
ax = sns.boxplot(x='n_source', y="gen_error", hue="$\\alpha$", data=errors, palette="Set3")  
ax.set(xlabel = '$n_P$', ylabel = '$E \mathcal{E}_Q(\hat f)$')
legend = '$\alpha$ with \n bandwidth $n^{-\alpha}$'
ax.set_title('Plot for $E \mathcal{E}_Q(\hat f)$ with bandwidth $n_P^{-\\alpha}$')
ax.get_figure().savefig('D:/GitHub/Tarnsfer-learning/without_label/errors.pdf')



ax = sns.boxplot(x='n_source', y="prop_error", hue="$\\alpha$", data=errors, palette="Set3")  
ax.set(xlabel = '$n_P$', ylabel = '$|\hat \pi_Q - \pi_Q|$')
legend = '$\alpha$ with \n bandwidth $n^{-\alpha}$'
ax.set_title('Plot for $|\hat \pi_Q - \pi_Q|$ with bandwidth $n_P^{-\\alpha}$')
ax.get_figure().savefig('D:/GitHub/Tarnsfer-learning/without_label/prop_errors.pdf')
    


    
    
