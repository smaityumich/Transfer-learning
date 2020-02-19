import numpy as np
from kde-classifier import *


class WithoutLabelClassifier():
    
    
    def __init__(self, x_source = np.random.random((100,3)), y_source = np.random.binomial(1, 0.5, (100,)), x_target = np.random.random((100,3))):
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
        
    def _estimateProportionRatio(self, classifier = None, n_neighbors = 5):
        
        '''
        Estimates the target/source proportions for different classes
        param classifier: a generic classifier for the source distribution with invertible confusion matrix
        
        If no classifier is provided then the default classifier chosen would be k-NN classifier with number of neighbors n_neighbors
        param n_neighbors: number of neighbors if classifier is None; default 5
        
        
        return w: vector of target/source prop for different classes
        
        See Lipton et al, Detecting and Correlating label shift with black box predictors
        '''
        
        if classifier == None:
            neigh = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors).fit(self.x_source, self.y_source)
            classifier = lambda x: neigh.predict(x)
            
        self._sourceClassifier = classifier
        
        confusionMatrix = metrics.confusion_matrix(classifier(self.x_source),self.y_source)/self.m
        print(f'Confusion matrix: {str(confusionMatrix)}\n')
        propTarget = np.mean(classifier(self.x_target))
        print(f'Target proportion of class 1 with classifier for source: {propTarget}')
        xi = np.array([1-propTarget,propTarget])
        self.w = np.matmul(np.linalg.inv(confusionMatrix),xi)
        self.prop_target = self.w[1]*self.prop_source
        return self.w
    
    def _classify(self, x_classify = 'None', classifier = 'None', bandwidth = None, kernel = 'gaussian'):
        '''Classifier for the target distribution
        param: x_classify, the numpy list (shape (n, d)) of features to classify
        param: bandwidth float; default = 0.01
        If you want to fit with cross validation set bandwith to None
        param: kernel str; default = 'gaussian'
        '''
        
        
        if type(x_classify) is str:
            x_classify = self.x_target
        else:
            x_classify = np.array(x_classify)
            if len(x_classify.shape) != 2:
                raise TypeError('Shape of x_classify is not (n,d)')
            if x_classify.shape[1] != self.x_target.shape[1]:
                raise TypeError(f'Dimension of feature space is not correct. It must be {self.x_target.shape[1]}')
     
        self.x_classify = x_classify
        
        if bandwidth == None:
            bandwidths =  np.linspace(0.1, 2, 100)
            grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths}, cv = 5)
            grid.fit(self.x_source, self.y_source)
            self.bandwidth = grid.best_params_['bandwidth']
        else: 
            self.bandwidth = bandwidth
        kde = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        kde.fit(self.x_source, self.y_source)   
        if type(classifier) is str:
            self._estimateProportionRatio(classifier = lambda x: kde.predict(x))
        else:
            self._estimateProportionRatio(classifier = classifier)
        kde = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        kde.fit(self.x_source, self.y_source, weights= self.w)
        self._targetClassifier = lambda x: kde.predict(x)
        return self._targetClassifier(x_classify)
    


       


    

   
