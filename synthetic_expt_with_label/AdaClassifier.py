# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:08:39 2019

@author: Subha Maity
"""

import numpy as np
import scipy.stats as st
from sklearn.neighbors import KNeighborsClassifier

class AdaClassifier():
    
    """Class for adaptive classifier. Here, the bandwidth r 
    is chosen adaptively via Lepski type algorithm"""
    
    def __init__(self, x_source, y_source, x_target, y_target, delta = 0.01):
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
        self.success = np.mean(self.y_target)
        
    def distances_(self, x):
        self.distance0 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x0)))
        self.distance1 = np.array(list(map(lambda y: np.linalg.norm(x-y, ord=2), self.x1)))
        
    def density_ratio(self, r):
        self.density0 = len(np.where(self.distance0<=r)[0])/self.n0
        self.density1 = len(np.where(self.distance1<=r)[0])/self.n1
        self.event = (self.density1 >= 72*self.alpha)
        if self.event:
            self.ratio = self.density0/self.density1
        else:
            self.ratio = 0
            
    def adaptation_(self, x):
        self.distances_(x)
        r = np.sqrt(self.d)
        while True:
            self.density_ratio(r)
            odds = (1-self.success)/self.success
            signal = np.absolute(odds*self.ratio - 1)
            if self.event:
                variance = odds*np.sqrt(24*self.alpha/self.density1)
            else:
                variance = odds/np.sqrt(3)
            SNR = signal/variance
            SNR = SNR**2
            if SNR < (self.d+3)*np.log(self.n0+self.n1):
                break
            else:
                r = r/2
        self.r = 2*r
        
    def predict_one(self, x):
        """Function for prediction of one instance
        :param x: instance to be predicted
        :return: class level, 0 or 1"""
        self.adaptation_(x)
        #self.r = self.d**self.d/((self.m)**(1/(2+self.d)))
        odds = (1-self.success)/self.success
        self.density_ratio(self.r)
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
    
    
class SimulateTransferLearning:
    
    
    def __init__(self, d = 3):
        self.d = d
        self.scale = 1
        
    def bayes_classifier(self, x):
        if self.density == "uniform":
            if self.p_target > 0.5:
                return 1
            else:
                return 0
        elif self.density == "truncated_normal":
            rv = st.norm(loc = 0, scale = self.scale)
            ratio = (1-self.p_target)/self.p_target
            for i in range(len(x)):
                ratio = ratio*rv.pdf(x[i])/rv.pdf(x[i]-1)
            if ratio < 1:
                return 1
            else:
                return 0
            
        elif self.density == "triangular":
            ratio = (1-self.p_target)/self.p_target
            for i in range(len(x)):
                ratio = ratio/(1.5 - np.absolute(2*x[i] -1))
            if ratio < 1:
                return 1
            else:
                return 0
            
        elif self.density == "trun_exp":
            ratio = (1-self.p_target)/self.p_target
            ratio = ratio/self.trun_exp_density(x)
            if ratio < 1:
                return 1
            else:
                return 0
            
        else:
            raise ValueError("Wrong input for density")
            
            
            
    def rv0(self):
        return np.random.random((self.d,))
    
    
    def rv1(self):
        pi = 0.5 #mixture proportion of uniform
        x = np.random.random((self.d,))
        y = np.random.triangular(0,0.5,1,(self.d,))
        z = np.random.binomial(1,pi,1)
        if z == 1:
            y = x
        return y
    def trun_exponential(self):
        x = np.zeros(self.d)
        for i in range(self.d):
            y = np.random.exponential(0.25)
            while y > 1:
                y = np.random.exponential(0.25)
                
            x[i] = y
            
        return x
    def trun_exp_density(self, x):
        r = 4/(1-np.exp(-4))
        density = (r**self.d)*np.exp(-4*np.linalg.norm(x, ord = 1))
        return density
        
            
            
        
        
    def classification(self, n = 100,  p = 0.75, density = 'uniform'):
        """Function for generating data in classification set-up
        :param n_source: sample size of source population
        :param n_target: sample size of target population
        :param p_source: float; probability of success for source distribution
        :param p_target: float; probability of success for target distribution
        :param density: string; should be "uniform" or "turncated_normal"
        :return: (x_source,y_source,x_target, y_target); 
            x_source, x_target: numpy array of covariates
            y_source, y_target: numpy array of class levels
        """
        self.p_target = p
        self.density = density
        
        if density == "uniform":
            x = np.random.random((n,self.d))
            y = np.random.binomial(1, p, n)
            return x, y
        
        elif density == "truncated_normal":
            y = np.random.binomial(1, p, n)
            rv0 = st.norm(loc = 0, scale = self.scale)
            rv1 = st.norm(loc = 1, scale = self.scale)
            x = np.zeros((n, self.d))
            for i in range(len(y)):
                if y[i] == 1:
                    rv = rv1
                else:
                    rv = rv0
                u = np.zeros(self.d)
                for j in range(self.d):
                    while True:
                        r = rv.rvs()
                        if abs(r) < 1:
                            break
                    u[j] = np.absolute(r)
                x[i] = u
            return x, y, np.array(list(map(self.bayes_classifier,x)))
        
        elif density == "triangular":
            y = np.random.binomial(1, p, n)
            x = np.zeros((n, self.d))
            for i in range(len(y)):
                if y[i] == 1:
                    x[i] = self.rv1()
                else:
                    x[i] = self.rv0()
            return x, y, np.array(list(map(self.bayes_classifier,x)))
        
        elif density == "trun_exp":
            y = np.random.binomial(1, p, n)
            x = np.zeros((n, self.d))
            for i in range(len(y)):
                if y[i] == 1:
                    x[i] = self.trun_exponential()
                else:
                    x[i] = self.rv0()
            return x, y, np.array(list(map(self.bayes_classifier,x)))
        
        else:
            raise ValueError("Invalid input for density. See help(SimulateTransferLearning.target_shift_)")
        
        
    def target_shift_(self, n_source = 1000, n_target = 50,  p_source = 0.5, p_target = 0.75, density = 'uniform'):
        """Function for generating data in target shift set-up
        :param n_source: sample size of source population
        :param n_target: sample size of target population
        :param p_source: float; probability of success for source distribution
        :param p_target: float; probability of success for target distribution
        :param density: string; should be "uniform", "turncated_normal" or "triangular"
        :return: (x, y, bayes_classifier); 
            x: numpy array of covariates
            y: numpy array of class levels
            bayes_classifiers_classifier: numpy array of bayes classification levels
        """
        
        if density == "uniform":
            x_source = np.random.random((n_source,self.d))
            x_target = np.random.random((n_target,self.d))
            y_source = np.random.binomial(1, p_source, n_source)
            y_target = np.random.binomial(1, p_target, n_target)
            return x_source, y_source, x_target, y_target
        
        elif density == "truncated_normal":
            y_source = np.random.binomial(1, p_source, n_source)
            y_target = np.random.binomial(1, p_target, n_target)
            rv0 = st.norm(loc = 0, scale = self.scale)
            rv1 = st.norm(loc = 1, scale = self.scale)
            x_source = np.zeros((n_source, self.d))
            x_target = np.zeros((n_target, self.d))
            for i in range(len(y_source)):
                if y_source[i] == 1:
                    rv = rv1
                else:
                    rv = rv0
                u = np.zeros(self.d)
                for j in range(self.d):
                    while True:
                        r = rv.rvs()
                        if abs(r) < 1:
                            break
                    u[j] = np.absolute(r)
                x_source[i] = u
            
            for i in range(len(y_target)):
                if y_source[i] == 1:
                    rv = rv1
                else:
                    rv = rv0
                u = np.zeros(self.d)
                for j in range(self.d):
                    while True:
                        r = rv.rvs()
                        if np.absolute(r) < 1:
                            break
                    u[j] = np.absolute(r)
                x_target[i] = u
            
            return x_source, y_source, x_target, y_target
        
        
        
        elif density == "triangular":
            y_source = np.random.binomial(1, p_source, n_source)
            y_target = np.random.binomial(1, p_target, n_target)
            x_source = np.zeros((n_source, self.d))
            x_target = np.zeros((n_target, self.d))
            for i in range(len(y_source)):
                if y_source[i] == 1:
                    x_source[i] = self.rv1()
                else:
                    x_source[i] = self.rv0()
                    
            for i in range(len(y_target)):
                if y_target[i] == 1:
                    x_target[i] = self.rv1()
                else:
                    x_target[i] = self.rv0()
                    
            return x_source, y_source, x_target, y_target
        
        
        elif density == "trun_exp":
            y_source = np.random.binomial(1, p_source, n_source)
            y_target = np.random.binomial(1, p_target, n_target)
            x_source = np.zeros((n_source, self.d))
            x_target = np.zeros((n_target, self.d))
            for i in range(len(y_source)):
                if y_source[i] == 1:
                    x_source[i] = self.trun_exponential()
                else:
                    x_source[i] = self.rv0()
                    
            for i in range(len(y_target)):
                if y_target[i] == 1:
                    x_target[i] = self.trun_exponential()
                else:
                    x_target[i] = self.rv0()
                    
            return x_source, y_source, x_target, y_target
        
        else:
            raise ValueError("Invalid input for density. See help(SimulateTransferLearning.target_shift_)")
        
    
            
num_points = 50    
            
def get_data(key):
    s = SimulateTransferLearning()
    x_source, y_source, x_target, y_target = s.target_shift_(n_source = int(key[1]), p_target=key[0], density = "triangular")
    x, y, bayes = s.classification(p = key[0], n = num_points,  density='triangular')
    return [(key, [x_source, y_source, x_target, y_target, x[i], bayes[i]]) for i in range(len(x))]


def classify(y):
    key, data = y
    cl = AdaClassifier(data[0],data[1],data[2],data[3])
    predict = cl.predict_one(data[4])
    predict_knn = cl.knn_(data[2],data[3],data[4])
    return key, [np.absolute(data[5]-predict), np.absolute(data[5]-predict_knn)]

def classify_list(y_list):
    return [classify(y) for y in y_list]




p_target = [0.7]
n_source = [100, 200, 400, 800]
ITER = range(10)
par = np.meshgrid(p_target, n_source ,ITER)
par = np.array(par)
par = par.reshape((3,-1))
par = par.T
par = [tuple(y) for y in par]







def vec_sum(x,y):
    return [(x[i]+y[i]) for i in range(len(x))]

import functools     
        

data = map(get_data, par)
error = map(classify_list, data)
error_list = list(error)

def generalized_error(y):
    key = y[0][0]
    value = [z[1] for z in y]
    gen_error = functools.reduce(vec_sum, value, [0,0])
    gen_error = [y/num_points for y in gen_error]
    
    return key, gen_error


gen_error = map(generalized_error, error_list)
gen_error = list(gen_error)
gen_error = list(map(lambda x: (x[0][:2],x[1]),gen_error))
gen_error









        
        
    

