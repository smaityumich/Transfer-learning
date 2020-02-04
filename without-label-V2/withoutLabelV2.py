import numpy as np
from sklearn import metrics, neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    w : numpy vector (2,)
        the inflation proportions for different classes
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y, weights = [1,1]):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [neighbors.KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        weights = np.array(weights)
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets] + np.log(weights)
        
        
    def predict_proba(self, X):
        self.logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(self.logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]




class WithoutLabelV2():
    
    
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
        
        
    def _estimateProportionRatio(self, classifier = None, n_neighbors = 5):
        
        '''
        Estimates the target/source proportions for different classes
        param classifier: a generic classifier for the source distribution with invertible confusion matrix
        
        If no classifier is provided then the default classifier chosen would be k-NN classifier with number of neighbors n_neighbors
        param n_neighbors: number of neighbors if classifier is None; default 5
        
        
        return w: vector of target/source prop for different classes
        
        See Lipton et al, Detecting and Correlating label shift with balck box predictors
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
        return self.w
    
    def _classify(self, x_classify = None, bandwidth = None, kernel = 'gaussian'):
        '''Classifier for the target distribution
        param: x_classify, the numpy list (shape (n, d)) of features to classify
        param: bandwidth float; default = 0.01
        If you want to fit with cross validation set bandwith to None
        param: kernel str; default = 'gaussian'
        '''
        
        
        if x_classify == None:
            x_classify = self.x_target
        else:
            x_classify = np.array(x_classify)
            if len(x_classify.shape) != 2:
                raise TypeError('Shape of x_classify is not (n,d)')
            if x_classify.shape[1] != self.x_target.shape[1]:
                raise TypeError(f'Dimension of feature space is not correct. It must be {self.x_target.shape[1]}')
     
        self.x_classify = x_classify
        
        if bandwidth == None:
            bandwidths =  np.linspace(0.1, 2, 20)
            grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths}, cv = 5)
            grid.fit(self.x_source, self.y_source)
            self.bandwidth = grid.best_params_['bandwidth']
        else: 
            self.bandwidth = bandwidth
        kde = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        kde.fit(self.x_source, self.y_source)   
        self._estimateProportionRatio(classifier = lambda x: kde.predict(x))
        #self.x_classify = x_classify
        kde = KDEClassifier(bandwidth = self.bandwidth, kernel = 'gaussian')
        kde.fit(self.x_source, self.y_source, weights= self.w)
        self._targetClassifier = lambda x: kde.predict(x)
        return self._targetClassifier(x_classify)
    
    
    

class DataGenerator():
    
    def __init__(self, d = 5):
        self.d = d
        
    def _generateY(self, n = 100, prop = 0.5):
        self.prop = prop
        self.n = n
        self.y = np.random.binomial(1, self.prop, (self.n,))
        
    def _generateX(self, distance = 1):
        self.mu = distance/np.sqrt(self.d)
        f = lambda y : np.random.normal(loc = y*self.mu, scale = 1, size = (self.d,))  ## Generates data from N_d(mu, I_d) if label=1, else from N_d(0,I_d) if label=0
        self.x = [f(y) for y in self.y]
        
    def _bayesDecision(self, x):
        x = np.array(x)
        prior = np.log(self.prop/(1-self.prop))
        log_lik_ratio = 0.5*np.sum(x**2) - 0.5*np.sum((x-self.mu)**2)  ## Calculates log-likelihood ratio for normal model Y=1: N(mu, 1); Y=0: N(0,1)
        posterior = prior + log_lik_ratio
        return 0 if posterior<0 else 1
        
    def _bayesY(self):
        self.bayesLabel = [self._bayesDecision(x) for x in self.x]
        
    def _getData(self, n = 100, prop = 0.5, distance = 0.2):
        self._generateY(n, prop)
        self._generateX(distance)
        self._bayesY()
        return np.array(self.x), np.array(self.y), np.array(self.bayesLabel)


        
        
    
