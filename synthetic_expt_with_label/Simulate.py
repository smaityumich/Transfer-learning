# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:08:54 2019

@author: Subha Maity
"""
import numpy as np


class SimulateClassification():
    
    def __init__(self, d=3):
        self.d = d
        self.x0 = np.repeat(0.5, self.d)
        
    def prop_(self, x):
        return 0.25 + 2*(np.linalg.norm(x-self.x0, ord = 2)**2)/self.d
    
    def xP_(self):
        x = np.random.random((2,self.d))
        y = np.random.binomial(1, 6/7)
        z = np.random.binomial(1, 4*self.prop_(x[1])/3)
        w = np.zeros(self.d)
        accept = False
        while not accept:
            if y == 1:
                accept = True
                w = x[0]
            else:
                if z == 1:
                    accept = True
                    w = x[1]
                else:
                    x = np.random.random((2,self.d))
                    y = np.random.binomial(1, 6/7)
                    z = np.random.binomial(1, 4*self.prop_(x[1])/3)
        return w
        
        
        
    def simulateQ_(self, n=100):
        x = np.random.random((n,self.d))
        y = np.zeros(n)
        for i in range(n):
            y[i] = np.random.binomial(1, self.prop_(x[i]))
            
        return x, y
    
    def simulateP_(self, n=100):
        x = np.zeros((n,self.d))
        y = np.zeros(n)
        for i in range(n):
            x[i] = self.xP_()
            success = 7*self.prop_(x[i])/(5+2*self.prop_(x[i]))
            y[i] = np.random.binomial(1,success)
        return x, y
    def bayes_rule_(self, n=100):
        x = np.random.random((n,self.d))
        y = np.zeros(n)
        for i in range(n):
            if self.prop_(x[i]) > 0.5:
                y[i] = 1
        return x, y
    
    
    
s = SimulateClassification(3)
