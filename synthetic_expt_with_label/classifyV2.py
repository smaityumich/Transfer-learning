# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:23:46 2019

@author: Subha Maity
"""

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
    is chosen adaptively via Lepski type algorithm
    Version2 is compatable to pyspark"""
    
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
        self.x_target = x_target
        self.y_target = y_target
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
        
        
    def dist1_(self, x):
        self.distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord = 1), self.x0)))
        self.distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord = 1), self.x1)))
        
        
    def dist2_(self, x):
        self.distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord = 2), self.x0)))
        self.distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord = 2), self.x1)))
        
        
    def density_ratio(self, x, r):
        if self.kernel == "truncation":
            self.dist2_(x)
            s = 0
            for i in range(self.n0):
                if self.distance0[i] < r:
                    s+=1
            self.density0 = s/self.n0
            s = 0
            for i in range(self.n1):
                if self.distance1[i] < r:
                    s+=1
            self.density1 = s/self.n1
        #print([self.density0, self.density1])
        elif self.kernel == "gaussian":
            self.dist2_(x)
            self.density0 = np.mean(np.exp(-(self.distance0/r)**2)/((np.sqrt(2*np.pi)*r)**self.d))
            self.density1 = np.mean(np.exp(-(self.distance1/r)**2)/((np.sqrt(2*np.pi)*r)**self.d))
        elif self.kernel == "exp":
            self.dist2_(x)
            self.density0 = np.mean(np.exp(-(self.distance0/r))/(r**self.d))
            self.density1 = np.mean(np.exp(-(self.distance1/r))/(r**self.d))
            
        elif self.kernel == "cauchy":
            self.dist2_(x)
            self.density0 = np.mean(r/(2*np.pi*(r**2 + self.distance0**2)))
            self.density1 = np.mean(r/(2*np.pi*(r**2 + self.distance1**2)))
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
    
    def knn_(self, x):
        neigh = KNeighborsClassifier(n_neighbors= 5)
        neigh.fit(self.x_target, self.y_target)
        return neigh.predict([x])[0]
    
    
    
    
    
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
    
n_source = [50*2**x for x in list(range(10))]
ITER = range(40)
par = np.meshgrid(ITER, n_source)
par = np.array(par)
par = par.reshape((2,-1))
par = par.T
par = [tuple(y) for y in par]



num_points = 50   
            
def get_data(key):
    s = SimulateClassification(3)
    x_P, y_P = s.simulateP_(int(key[1]))
    x_Q, y_Q = s.simulateQ_(20)
    x_test, bayes = s.bayes_rule_(num_points)
    return [(key, [x_P, y_P, x_Q, y_Q, x_test[i], bayes[i]]) for i in range(len(x_test))]


def classify(y):
    key, data = y
    print(str(key))
    s = SimulateClassification(3)
    cl = AdaClassifier(data[0],data[1],data[2],data[3], kernel="cauchy")
    predict = cl.predict_one(data[4])
    predict_knn = cl.knn_(data[4])
    prop = np.absolute(s.prop_(data[4]) - 0.5)
    return key, [np.absolute(data[5]-predict)*prop, np.absolute(data[5]-predict_knn)*prop]

def classify_list(y_list):
    return [classify(y) for y in y_list]

   
    
    
