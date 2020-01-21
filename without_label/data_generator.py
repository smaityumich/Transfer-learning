# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:01:42 2020

@author: Subha Maity
"""

import numpy as np
import scipy.stats as st
import functools

class GeneratorClassification():
    
    def __init__(self, d = 3, covariate = 'normal', mu = 1):
        
        ## Debugging faulty dimension input
        try: 
            d = int(d)
        except: 
            raise TypeError('d can\'t be converted into integer.')
            
        self.d = d
        
        if covariate != 'normal' and covariate != 'exp':
            raise TypeError('Invalid covariate distribution input')
        else:
            self.covariate = covariate
            
        ## Debugging input of mu
        try:
            mu = float(mu)
        except:
            raise TypeError('Alternative mean can\'t be converted into float')
        
        self.mu = mu
        
    def _generateCovariate(self, y):
        
        if self.covariate == 'normal':
            z = np.random.normal(0, 1, self.d)
            if y == 0:
                return z
            else:
                return [u+self.mu for u in z]
            
        elif self.covariate == 'exp':
            z = np.random.exponential(1, self.d)
            if y == 0:
                return z
            else:
                return [u+self.mu for u in z]
            
        else:
            raise TypeError('Wrong distribution input')
            
            
    def _logdensityratio(self, x):
        ## Here x is a scalar
        if self.covariate == 'normal':
            return np.log(st.norm.pdf(x)) - np.log(st.norm.pdf(x, loc = self.mu))
        elif self.covariate == 'exp':
            return np.log(st.expon.pdf(x)) - np.log(st.expon.pdf(x, loc = self.mu))
            
    def _generate(self, n = 100,  prop = 0.5):
        
        ## Debugging faulty sample size input
        try: 
            n = int(n)
        except: 
            raise TypeError('n can\'t be converted into integer.')
            
        y = np.random.binomial(1, prop, n)
        x = [self._generateCovariate(y[i]) for i in range(n)]
        return np.array(x), np.array(y)
    
    def _bayesClassifier(self, x, prop_target):
        
        posteriorRatio = functools.reduce(lambda u,v: self._logdensityratio(u) + self._logdensityratio(v), x, 0) + np.log(prop_target) - np.log(1-prop_target)
        if posteriorRatio >= 0:
            return 1
        else:
            return 0
        