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
from densit_ratio import *

class ClassifierNoLabel():
    
    def __init__(self, x_source = np.zeros((1,3)), y_source = np.zeros(1), x_target = np.zeros((1,3)), kernel_type = 'normal'):
        
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
        if self.DensityRatio.d != x_target.shape[1]:
            raise TypeError('Dimension of sourse and target distribution doesn\'t match')
            
            
        ## Debugging kernel type
        if kernel_type != 'normal' and kernel_type != 'exp':
            raise TypeError('Invalid kernel type')
            
        self.kernel_type = kernel_type
        
        self.x_target = x_target
        self.prop_target = 0.5
        
        x0 = x_source[y_source == 0]
        x1 = x_source[y_source == 1]
        self.DensityRatio = DensityRatio(x0,x1)
        
    def _classify(self, x, h):
        '''Classify a point'''
        densityratio = self.DensityRatio._densityratio(x, h, self.kernel_type)
        odds_ratio = 1/self.prop_source - 1
        regfn = 1/(1+odds_ratio*densityratio)
        label = 0
        if regfn > 0.5:
            label = 1
        return label
    
    def _targetProp(self, h):
        '''To estimate proportion of success in target population'''
        targetlabel = [self._classify(u, h) for u in self.x_target]
        return np.mean(targetlabel)
    
    def _targetPropEstimate(self, h, kernel_type = 'normal', max_step = 100, threshold = 1e-2):
        '''Iterative algo to find prop of success on target population'''
        step = 0
        error = 1
        while error>threshold:
            targetprop = self._targetProp(h)
            error = np.absolute(self.prop_target-targetprop)
            step += 1
            self.prop_target = targetprop
            if step == max_step:
                break