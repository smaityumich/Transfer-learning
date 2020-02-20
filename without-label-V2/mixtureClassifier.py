import numpy as np
from kdeClassifier import *

class MixtureClassifier():
"""
For a mixture proportion and a predictior x it predicts the Q(Y|x) as (1-mixture)*(estimated success prob for P-data) + mixture*(estimated success prob for Q-data) and decides the label according to popular voting scheme

The individual success probabilities are estimated using kernel density, where the smoothing parameter is chosen according to cross validation
"""
    def __init__(self, mixture = 0.5, kernel = 'gaussian'):
        self.kernel = kernel

    def fit(self, x_source = np.random.random((100,3)), y_source = np.random.binomial(1, 0.5, (100,)), x_target = np.random.random((100,3)), y_target = np.random.binomial(1, 0.5, (100,))):
        
        self._classifier_source = KDEClassifierOptimalParameter()
        self._classifier_source.fit(x_source, y_source)

        self._classifier_target = KDEClassifierOptimalParameter()
        self._classifier_target.fit(x_target, y_target)


