import numpy as np
from sklearn import metrics, neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


class WithLabelClassifier(BaseEstimator, ClassifierMixin):

    
    def __init__(self, bandwidth = 1.0, kernel = 'gaussian'):
        
        self.bandwidth = bandwidth
        self.kernel = kernel


    def _sourcedata(self, x_source = np.random.random((100,3)), y_source = np.random.binomial(1, 0.5, (100,))):   
        
        x_source = np.array(x_source)
        y_source = np.array(y_source)


        # Checking shape consistency
        if len(x_source.shape) != 2:
            raise TypeError('x_source is not an array fo shape (n,d)')
        if len(y_source.shape) != 1:
            raise TypeError('y_source is not an array fo shape (n,)')
        
        self.x_source = x_source
        self.y_source = y_source


        


    def fit(self, x_target = np.random.random((100,3)), y_target = np.random.binomial(1, 0.5, (100,))):
        '''
        __init__: To store all the data in a class
        param 
        x_source: numpy array (n,d) of features in source distribution
        y_source: numpy array (n,) of labels in source distribution
        x_target: numpy array (n,d) of features in target distribution
        y_target: numpy array (n,) of labels in target distribution

        
        Stores the class variables 
        m: # source data points
        n: # target data points
        d: # feature dimension
        x_source, y_source, x_target, y_target
        '''
     
        x_target = np.array(x_target)
        y_target = np.array(y_target)
        
        # Checking shape consistency
        if len(x_target.shape) != 2:
            raise TypeError('x_target is not an array fo shape (n,d)')
        if len(y_target.shape) != 1:
            raise TypeError('y_target is not an array fo shape (n,)')
           

        # Checking dimension consistency
        if self.x_source.shape[1] != x_target.shape[1]:
            raise TypeError('Dimension don\'t match for source and target features')
            
        m, self.d = self.x_source.shape
        self.n, _ = x_target.shape
        self.n += m
        x, y = np.concatenate((self.x_source, x_target)), np.concatenate((self.y_source, y_target))
        self.prop_target = np.mean(y_target)
        weights = np.array([1-self.prop_target, self.prop_target])
        self.logpriors_ = np.log(weights)
        self.classes = np.array([0, 1])
        training_sets = [x[y == i] for i in [0, 1]]
        self.models_ = [neighbors.KernelDensity(bandwidth=self.bandwidth, kernel = self.kernel).fit(xi) for xi in training_sets]



    def predict(self, x = np.random.normal(0, 1, (20, 3))):
        
        self.logprobs = np.array([model.score_samples(x) 
                             for model in self.models_]).T
        result = np.exp(self.logprobs +  self.logpriors_)
        posterior = result / result.sum(1, keepdims = True)
        return self.classes[np.argmax(posterior, 1)]
        



class WithLabelOptimalClassifier():

    def __init__(self, kernel = 'gaussian'):
        self.kernel = kernel


    def fit(self, x_source = np.random.random((100,3)), y_source = np.random.binomial(1, 0.5, (100,)), x_target = np.random.random((100,3)), y_target = np.random.binomial(1, 0.5, (100,))):

        bandwidths = np.linspace(0.1, 2, 100)
        grid = GridSearchCV(WithLabelClassifier(), {'bandwidth': bandwidths}, cv = 5)
        grid._sourcedata(x_source, y_source)
        grid.fit(x_target, y_target)
        self.bandwidth = grid.best_params_['bandwidth']
        self._classifier = WithLabelClassifier(bandwidth = self.bandwidth, data = (x_source, y_source)).fit(x_target, y_target)


    def predict(self, x = np.random.normal(0, 1, (10, 3))):
        return self._classifier.predict(x)
 
