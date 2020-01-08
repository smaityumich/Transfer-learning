# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:07:23 2019

@author: Subha Maity
"""

import numpy as np
import scipy as sc
from sklearn.neighbors import KNeighborsClassifier

class AdaClassifier():
    
    """Class for adaptive classifier. Here, the bandwidth r 
    is chosen adaptively via Lepski type algorithm"""
    
    def __init__(self, x_source, y_source, x_target, y_target, delta = 0.01, kernel = "gaussian"):
        """Function for initializing the class AdaClassifier
        :param x_source: a (n_s, d) numpy array for covariates of source
        :param y_source: a (n_s, ) numpy array for class levels of source
        :param x_source: a (n_t, d) numpy array for covariates of target
        :param y_source: a (n_t, ) numpy array for class levels of target
        :param delta: exception probability, default 0.01
        """
        self.y_target = y_target
        x_all = np.concatenate((x_source, x_target), axis = 0)
        y_all = np.concatenate((y_source , y_target))
        self.d = x_source.shape[1]
        self.x0 = x_all[np.where(y_all==0)]
        self.x1 = x_all[np.where(y_all==1)]
        self.n0 = len(self.x0)
        self.n1 = len(self.x1)
        self.m = min([self.n0,self.n1])
        self.alpha = ((self.d+1)*np.log(2*self.m))/self.m
        self.success = 5/12#np.mean(self.y_target)
        self.volunit_ = (np.pi**(self.d/2)/sc.special.gamma(self.d/2+1))
        self.kernel = kernel
        
    def distances_(self, x):
        self.distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x0)))
        self.distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x1)))
        #print(self.distance0)
        
    def density_ratio(self, x, r):
        if self.kernel == "truncation":
            distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x0)))
            distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x1)))
            s = 0
            for i in range(self.n0):
                if distance0[i] < r:
                    s+=1
            self.density0 = s/self.n0
            s = 0
            for i in range(self.n1):
                if distance1[i] < r:
                    s+=1
            self.density1 = s/self.n1
        #print([self.density0, self.density1])
        elif self.kernel == "gaussian":
            distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x0)))
            distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x1)))
            self.density0 = np.mean(np.exp(-(distance0/r)**2)/((np.sqrt(2*np.pi)*r)**self.d))
            self.density1 = np.mean(np.exp(-(distance1/r)**2)/((np.sqrt(2*np.pi)*r)**self.d))
        elif self.kernel == "exp":
            distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=1), self.x0)))
            distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=1), self.x1)))
            self.density0 = np.mean(np.exp(-(distance0/r))/(r**self.d))
            self.density1 = np.mean(np.exp(-(distance1/r))/(r**self.d))
            
        elif self.kernel == "cauchy":
            distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x0)))
            distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x1)))
            self.density0 = np.mean(r/(2*np.pi*(r**2 + distance0**2)))
            self.density1 = np.mean(r/(2*np.pi*(r**2 + distance1**2)))
        self.event = (self.density1 >= self.alpha/10)
        if self.event:
            self.ratio = self.density0/self.density1
        else:
            self.ratio = 0
            
    def adaptation_(self, x):
        r = 0.5
        step = 1
        while True:
            self.density_ratio(x, r)
            odds = (1-self.success)/self.success
            signal = np.absolute(odds*self.ratio - 1)
            #vol = self.volunit_*(r**self.d)
            if self.event:
                variance = odds*np.sqrt(self.alpha/self.density1)
            else:
                variance = odds/np.sqrt(3*24)
            SNR = signal/variance
            #print("SNR : "+str(SNR)+'\n')
            SNR = SNR**2
            if SNR > 3 or self.density1 < 5/self.n1:
                break
            else:
                r = r/(1+1/step)
                step = step + 1
        self.r = r
        #print("r: "+str(self.r)+'\n')
        
    def predict_one(self, x):
        """Function for prediction of one instance
        :param x: instance to be predicted
        :return: class level, 0 or 1"""
        self.adaptation_(x)
        #self.r = self.d**self.d/((self.m)**(1/(2+self.d)))
        odds = (1-self.success)/self.success
        self.density_ratio(x, self.r)
        posterior_odds = odds*self.ratio
        if posterior_odds<1:
            return 1
        else:
            return 0
        
    def predict_(self, x):
        """For prediction of more than one instances
        :param x: a (n,d) numpy array of instances
        :return: a numpy array of class levels"""
        return np.array(list(map(self.predict_one, x)))
    
    def knn_(self, x_train, y_train, x):
        neigh = KNeighborsClassifier(n_neighbors= 5)
        neigh.fit(x_train, y_train)
        return neigh.predict([x])[0]