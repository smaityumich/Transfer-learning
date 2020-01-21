# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:15:21 2020

@author: Subha Maity
"""

import numpy as np
import functools as ft
from scipy import stats

class KernelDensityEstimator():
    
    '''Class for calculating density ratio with kernel density estimator
    Adapted only for normal and exponential kernel'''
    
    def __init__(self, x, y):
        try:
            x = np.array(x)
            y = np.array(y)
        except:
            raise TypeError('Can\'t convert it to numpy array')
            
        # Check for 2d array
        if x.ndim == 1:
            x.reshape((len(x),1))
        elif x.ndim > 2:
            raise TypeError('x is not a 2d array')
            
            
        if y.ndim == 1:
            y.reshape((len(y),1))
        elif y.ndim > 2:
            raise TypeError('y is not a 2d array')
        
        # Equality of dimension check
        if x.shape[1] != y.shape[1]:
            raise ValueError('Arrays are of different dimensions')
            
        # Store the arrays
        self.x = x
        self.y = y
        self.m, self.d =  x.shape
        self.n = y.shape[0]
        
        
    def _kde(self, kernel, arr, u):
        try:
            u = np.array(u)
        except:
            raise TypeError('Can\'t transform into numpy array')
            
        if len(u) != self.d:
            raise ValueError('Dimension don\'t match')
        else:
            kernels = map(lambda x: kernel(x-u), arr)
            return np.mean(list(kernels))
        
    def _normalkernel(self, x, h):
        return ft.reduce(lambda a, b: stats.norm.pdf(a,loc = 0, scale = h)*stats.norm.pdf(b, loc = 0,  scale = h), x, 1)
    
    def _expkernel(self, x, h):
        return ft.reduce(lambda a, b: stats.expon.pdf(a, loc = 0, scale = h)*stats.expon.pdf(b, loc = 0, scale = h), x, 1)
    
    def _densities(self, u, h, kernel_type = "normal"):
        if kernel_type == "normal":
            return self._kde(lambda a: self._normalkernel(a,h), self.x, u), self._kde(lambda a: self._normalkernel(a,h), self.y, u)
        elif kernel_type == "exp":
            return self._kde(lambda a: self._expkernel(a,h), self.x, u), self._kde(lambda a: self._expkernel(a,h), self.y, u)
        else: 
            raise TypeError("Invalid Kernel")
        
        
        
