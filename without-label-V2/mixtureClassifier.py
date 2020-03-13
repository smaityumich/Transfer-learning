import numpy as np
from kdeClassifier import *
from sklearn.base import BaseEstimator
from sklearn import base
from sklearn.model_selection import KFold, ParameterGrid
from multiprocessing import Pool
import multiprocessing as mp


class MixtureClassifier(BaseEstimator):
    '''
    For a mixture proportion and a predictior x it predicts the Q(Y|x) as (1-mixture)*(estimated success prob for P-data) + mixture*(estimated success prob for Q-data) and decides the label according to popular voting scheme

    The individual success probabilities are estimated using kernel density, where the smoothing parameter is chosen according to cross validation
    '''
    def __init__(self, mixture = 0.5, kernel = 'gaussian', workers = 1):
        self.kernel = kernel
        self.mixture = mixture
        self.workers = workers

    def fit(self, source_classifier,  x_target, y_target):
        self.source_classifier = source_classifier
        self.prop = np.mean(y_target) 
        cl = KDEClassifierOptimalParameter(workers = self.workers)
        cl.fit(x_target, y_target)
        self.target_classifier = cl._classifier
 
    def predict_proba(self, x): 
        '''
        Both source and target classifier must have predict_proba method for predicting the class probabilities 
        '''
        return (1-self.mixture)*self.source_classifier.predict_proba(x) + self.mixture*self.target_classifier.predict_proba(x)
    

    def predict(self, x):
        if self.prop == 0:
            return [0 for _ in x]
        elif self.prop == 1:
            return [1 for _ in x]
        else:
            prob = self.predict_proba(x)
            classes = np.array([0, 1])
            return classes[np.argmax(prob, 1)]




class OptimalMixtureClassifier():

    def __init__(self, cv = 5, nodes = 1):
        self.cv = 5
        self.workers = nodes


    def unit_work(self, args):
        method, arg, data = args
        source_classifier, x_target, y_target = data
        kf = KFold(n_splits=self.cv)
        errors = np.zeros((self.cv,))

        for index, (train_index, test_index) in enumerate(kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            method.fit(source_classifier, x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_test-y_pred)**2)

        return {'arg': arg, 'error': np.mean(errors)}



    def fit(self, x_source, y_source, x_target, y_target, cv = 5):

        cl = KDEClassifierOptimalParameter(workers = self.workers)
        cl.fit(x = x_source, y = y_source)
        source_classifier = cl._classifier


        cl = MixtureClassifier(workers = self.workers)
        params = {'mixture': np.linspace(0, 1, 20)}
        par_list = list(ParameterGrid(params))
        models = [base.clone(cl).set_params(**arg) for arg in par_list]
        data = source_classifier, x_target, y_target
        datas = [data for _ in range(len(par_list))]
        list_errors = list(map(self.unit_work, zip(models, par_list, datas)))
        error_list = np.array([s['error'] for s in list_errors])
        self.mixture = list_errors[np.argmin(error_list)]['arg']['mixture']
        self.classifier = MixtureClassifier(mixture = self.mixture)
        self.classifier.fit(source_classifier, x_target, y_target)

    def predict(self, x):
        return self.classifier.predict(x)

