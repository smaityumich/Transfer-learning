# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:24:23 2020

@author: Subha Maity
"""
import sys
sys.path.insert(1, 'D:/GitHub/Tarnsfer-learning/without_label/')

## Change the path to local directory:
## sys.path.insert(1, '${pwd}')

import numpy as np
import sklearn.metrics
import scipy
from sklearn.neighbors import KernelDensity

class ClassifierNoLabel():
    
    def __init__(self, bandwidth = 0.1, kernel_type = 'gaussian'):
        
        self.bandwidth = bandwidth
        self.kernel = kernel_type
        
    
    def _data(self, x_source = np.zeros((1,3)), y_source = np.zeros(1), x_target = np.zeros((1,3))):
        
        ## Converting to numpy array
        try:
            x_target = np.array(x_target)
        except:
            raise TypeError('x_target can\'t be converted into numpy array')
            
        ## Checking dimension
        if x_target.ndim == 1:
            x_target.reshape((len(x_target),1))
        elif x_target.ndim > 2:
            raise TypeError('x_target is not 1d or 2d array')
            
        ## Check for sourse data
        if len(x_source) != len(y_source):
            raise TypeError('Number of features and labels does not match in sourse distribution')
            
        ## Checking dimension consistency
        if x_source.shape[1] != x_target.shape[1]:
            raise TypeError('Dimension of sourse and target distribution doesn\'t match')
        
        self.x_target = x_target
        self.prop_target = 0.5
        self.x_source = x_source
        self.y_source = y_source
        self.prop_source = np.mean(y_source)
        
        x0 = x_source[y_source == 0]
        x1 = x_source[y_source == 1]
        
        self.KDE0 = KernelDensity(bandwidth= self.bandwidth, kernel = self.kernel).fit(x0)
        self.KDE1 = KernelDensity(bandwidth= self.bandwidth, kernel= self.kernel).fit(x1)
        
        
        
    def _classify(self, x, y):
        if x > y:
            return 0
        else:
            return 1
        
        
    def _classifySource(self, x):
        '''Classify a point for source data'''
        log_density0, log_density1 = self.KDE0.score_samples(x) + np.log(1-self.prop_source), self.KDE1.score_samples(x) + np.log(self.prop_source)
        label = [self._classify(log_density0[_],log_density1[_]) for _ in range(len(log_density0))]
        return np.array(label)
    
    def _targetProp(self):
        '''To estimate proportion of success in target population'''
        targetlabel = self._classifySource(self.x_target)
        return np.mean(targetlabel)
    
    def _targetPropEstimate(self, max_step = 100, threshold = 1e-2):
        '''Iterative algo to find prop of success on target population
        Credit: Yuekai Sun'''
        step = 0
        error = 1
        while error>threshold:
            targetprop = self._targetProp()
            error = np.absolute(self.prop_target-targetprop)
            step += 1
            self.prop_target = targetprop
            if step == max_step:
                break
            
            
    def _targetPropBlackbox(self):
        '''Uses black-box predictor to detect label shift.
        See: Lipton et al. Detecting and Correcting for Label Shift with Black Box Predictors (2018)'''
        
        predict_source = self._classifySource(self.x_source)  #predicted source lables
        predict_target = self._classifySource(self.x_target)  #predicted target labels
        confusion_mx = sklearn.metrics.confusion_matrix(self.y_source, predict_source) #confusion matrix
        y_hat_target = np.mean(predict_target)
        mu_hat = [1-y_hat_target, y_hat_target]
        w_hat = np.matmul(scipy.linalg.inv(confusion_mx), mu_hat)
        mu_estimated = np.matmul(np.diag([1-np.mean(predict_source),np.mean(predict_source)]), w_hat)
        self.prop_target = mu_estimated[1]
        return w_hat, mu_estimated
    
    
    def _classifyTarget(self, x):
        '''Classify a point for target data'''
        log_density0, log_density1 = self.KDE0.score_samples(x) + np.log(1-self.prop_target), self.KDE1.score_samples(x) + np.log(self.prop_target)
        label = [self._classify(log_density0[_],log_density1[_]) for _ in range(len(log_density0))]
        return np.array(label)
        