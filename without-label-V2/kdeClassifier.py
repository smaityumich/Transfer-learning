import numpy as np
from sklearn import metrics, neighbors, base
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold
from multiprocessing import Pool


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
        self.prop = np.mean(y)
        self.priors_ = np.multiply(np.array([1-self.prop, self.prop]) , weights)
        
        
    def predict_proba(self, X):
        self.logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.multiply(np.exp(self.logprobs), self.priors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        if self.prop == 0:
            return [0 for _ in x]
        elif self.prop == 1:
            return [1 for _ in x]
        else:
            return self.classes_[np.argmax(self.predict_proba(X), 1)]




class KDEClassifierOptimalParameter():

    '''
    Finds the smoothness parameter optimally using cross-vaidation
    '''

    def __init__(self, bandwidth = None, cv = 5, workers = 1):
        self.bandwidth  = bandwidth
        self.cv = cv
        self.workers = workers



    def unit_work(self, args):
        method, arg , data = args
        x, y = data
        kf = KFold(n_splits = self.cv)
        errors = np.zeros((self.cv, ))

        for index, (train_index, test_index) in enumerate(kf.split(x)):
            x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
            method.fit(x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_pred-y_test)**2)

        return {'arg': arg, 'error': np.mean(errors), 'errors': errors}






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
            self.bandwidth = float(self.bandwidth)
        except:
            bandwidths =  np.linspace(0.1, 2, 20)
            cl = KDEClassifier()
            params = {'bandwidth': bandwidths}
            par_list = list(ParameterGrid(params))
            models = [base.clone(cl).set_params(**arg) for arg in par_list]
            data = x, y
            datas = [data for _ in range(len(par_list))]
            args = zip(models, par_list, datas)
            
            if self.workers == 1:
                self.list_errors = list(map(self.unit_work, args))

            else:
                with Pool(self.workers) as pool:
                    self.list_errors = pool.map(self.unit_work, zip(models, par_list, datas))

            error_list = np.array([s['error'] for s in self.list_errors])
            self.bandwidth = self.list_errors[np.argmin(error_list)]['arg']['bandwidth']
 
        self._classifier = KDEClassifier(bandwidth = self.bandwidth)
        self._classifier.fit(self.x, self.y)

    def predict_proba(self, x):
        return self._classifier.predict_proba(x)
     
    def predict(self, x):
        return self._classifier.predict(x)

 
