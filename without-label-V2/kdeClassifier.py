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




class KDEClassifierOptimalParameter():

    '''
    Finds the smoothness parameter optimally using cross-vaidation
    '''

    def __init__(self, bandwidth = None):
        self.bandwidth  = bandwidth

    def fit(self, x = np.random.random((100,3)), y = np.random.binomial(1, 0.5, (100,))):
        x = np.array(x)
        y = np.array(y)
        
        # Checking shape consistency
        if len(x.shape) != 2:
            raise TypeError('x is not an array fo shape (n,d)')
        if len(y.shape) != 1:
            raise TypeError('y is not an array fo shape (n,)')
            
            
        self.n, self.d = x.shape
        self.x, self.y = x, y
        try: 
            self.bandwith = float(self.bandwidth)
        except:
            bandwidths =  np.linspace(0.1, 2, 20)
            grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths}, cv = 5)
            grid.fit(self.x, self.y)
            self.bandwidth = grid.best_params_['bandwidth']
        self._classifier = KDEClassifier(bandwidth = self.bandwidth)
        self._classifier.fit(self.x, self.y)

    def predict_proba(self, x):
        return self._classifier.predict_proba(x)
     
    def predict(self, x):
        return self._classifier.predict(x)

 
