import numpy as np
from kdeClassifier import *
from sklearn.model_selection import KFold, ParameterGrid
from sklearn import base
from multiprocessing import Pool



class WithoutLabelClassifier():

    def __init__(self, kernel = 'gaussian', workers = 1, cv = 5):
        self.kernel = kernel
        self.workers = workers
        self.cv = cv


    def unit_work(self, args):
        method, arg , data = args
        x_source, y_source = data
        kf = KFold(n_splits = self.cv)
        errors = np.zeros((self.cv, ))

        for index, (train_index, test_index) in enumerate(kf.split(x_source)):
            x_train, x_test, y_train, y_test = x_source[train_index], x_source[test_index], y_source[train_index], y_source[test_index]
            method.fit(x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_pred-y_test)**2)

        return {'arg': arg, 'error': np.mean(errors), 'errors': errors}

    
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
            bandwidths =  np.linspace(0.1, 2, 20)
            cl = KDEClassifier()
            params = {'bandwidth': bandwidths}
            par_list = list(ParameterGrid(params))
            models = [base.clone(cl).set_params(**arg) for arg in par_list]
            data = x_source, y_source
            datas = [data for _ in range(len(par_list))]
            
            with Pool(self.workers) as pool:
                 self.list_errors = pool.map(self.unit_work, zip(models, par_list, datas))

            error_list = np.array([s['error'] for s in self.list_errors])
            
            self.bandwidth = self.list_errors[np.argmin(error_list)]['arg']['bandwidth']
        if bandwidth != None:
            self.bandwidth = bandwidth

        self._sourceClassifier = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        self._sourceClassifier.fit(self.x_source, self.y_source)

            
        
        confusionMatrix = metrics.confusion_matrix(self._sourceClassifier.predict(self.x_source),self.y_source)/self.m
        propTarget = np.mean(self._sourceClassifier.predict(self.x_target))
        xi = np.array([1-propTarget,propTarget])
        self.w = np.matmul(np.linalg.inv(confusionMatrix),xi)
        self.prop_target = self.w[1]*self.prop_source
        self._targetClassifier = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        self._targetClassifier.fit(self.x_source, self.y_source, weights= self.w)
        
    
    def predict(self, x):
        return self._targetClassifier.predict(x)


       


    

   
