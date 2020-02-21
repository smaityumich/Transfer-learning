import numpy as np
from kdeClassifier import *


class WithoutLabelClassifier():

    def __init__(self, kernel = 'gaussian'):
        self.kernel = kernel

    
    def fit(self, x_source = np.random.random((100,3)), y_source = np.random.binomial(1, 0.5, (100,)), x_target = np.random.random((100,3)), classifier = 'None', bandwidth = None):
        '''
        __init__: To store all the data in a class
        param x_source: numpy array (n,d) of features in source distribution
        param y_source: numpy array (n,) of labels in source distribution
        param x_target: numpy array (n,d) of features in target distribution
        
        Stores the class variables 
        m: # source data points
        n: # target data points
        d: # feature dimension
        x_source, y_source, x_target
        '''
     
        x_source = np.array(x_source)
        y_source = np.array(y_source)
        x_target = np.array(x_target)
        
        # Checking shape consistency
        if len(x_source.shape) != 2:
            raise TypeError('x_source is not an array fo shape (n,d)')
        if len(x_target.shape) != 2:
            raise TypeError('x_target is not an array fo shape (n,d)')
        if len(y_source.shape) != 1:
            raise TypeError('y_source is not an array fo shape (n,)')
            
        # Checking dimension consistency
        if x_source.shape[1] != x_target.shape[1]:
            raise TypeError('Dimension don\'t match for source and target features')
            
        self.m, self.d = x_source.shape
        self.n, _ = x_target.shape
        self.x_source, self.y_source, self.x_target = x_source, y_source, x_target
        self.prop_source = np.mean(y_source) 
        
        
        '''
        Estimates the target/source proportions for different classes
        param classifier: a generic classifier for the source distribution with invertible confusion matrix
        
        If no classifier is provided then the default classifier chosen would be k-NN classifier with number of neighbors n_neighbors
        param n_neighbors: number of neighbors if classifier is None; default 5
        
        
        return w: vector of target/source prop for different classes
        
        See Lipton et al, Detecting and Correlating label shift with black box predictors
        '''
        if bandwidth == None or classifier == 'None':
            bandwidths =  np.linspace(0.1, 2, 40)
            grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths}, cv = 5)
            grid.fit(self.x_source, self.y_source)
            self.bandwidth = grid.best_params_['bandwidth']
        if bandwidth != None:
            self.bandwidth = bandwidth

        self._sourceClassifier = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        self._sourceClassifier.fit(self.x_source, self.y_source)

            
        
        confusionMatrix = metrics.confusion_matrix(self._sourceClassifier.predict(self.x_source),self.y_source)/self.m
        #print(f'Confusion matrix: {str(confusionMatrix)}\n')
        propTarget = np.mean(self._sourceClassifier.predict(self.x_target))
        #print(f'Target proportion of class 1 with classifier for source: {propTarget}')
        xi = np.array([1-propTarget,propTarget])
        self.w = np.matmul(np.linalg.inv(confusionMatrix),xi)
        self.prop_target = self.w[1]*self.prop_source
        self._targetClassifier = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        self._targetClassifier.fit(self.x_source, self.y_source, weights= self.w)
        
    
    def predict(self, x):
        return self._targetClassifier.predict(x)


       


    

   
