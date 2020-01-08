# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:15:21 2020

@author: Subha Maity
"""

import numpy as np
import functools as ft
from scipy import stats

class DensityRatio():
    
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
        normal = stats.norm()
        return ft.reduce(lambda a, b: normal.pdf(a,scale = h)*normal.pdf(b, scale = h), x, 1)
    
    def _expkernel(self, x, h):
        exp = stats.expon()
        return tf.reduce(lambda a, b: exp.pdf(a, scale = h)*exp.pdf(b, scale = h), x, 1)
    
    def _densityratio(self, kernel_type = "normal", u, h):
        if kernel_type == "normal":
            
        
        
        
        
        
