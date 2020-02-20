from kdeClassifier import *
from withLabelV2 import *
from withoutLabelV2 import *
import numpy as np
from dataGenerator import *



class Experiments():

    def __init__(self, kernel = 'gaussian'):
        self.kernel = kernel

    def _getData(self, n_source = 500, n_target = 200, n_test = 200, prop_source = 0.5, prop_target = 0.8, dist = 0.8, d = 4):
        '''
        Generates data fro simulation purpose

        attributes:

        n_source: int, number of data-points in source data
        n_target: int, number of data-points in target data
        n_test: int, number of data-points in test data; distribution of test data is same as that of target data
        prop_source: float, prob of success in source data
        prop_target: float, prob of success in target data
        dist: distance of means between class conditional distributions 
        d: int, feature space dimension
        '''
        datageneretor = DataGenerator(d = d)
        self.data = dict()
        self.data['source-data'] = dict()
        s =  self.data['source-data']
        s['x'], s['y'] , _ = datageneretor._getData(n = n_source, prop=prop_source, distance=dist)

        self.data['target-data'] = dict()
        s = self.data['target-data'] 
        s['x'], s['y'] , _ = datageneretor._getData(n = n_target, prop=prop_target, distance=dist)

        self.data['test-data'] = dict()
        s = self.data['test-data'] 
        s['x'], s['y'], s['bayes'] = datageneretor._getData(n = n_test, prop = prop_target, distance = dist)

