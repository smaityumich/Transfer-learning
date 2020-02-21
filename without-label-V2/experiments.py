from kdeClassifier import *
from withLabelV3 import *
from withoutLabelV3 import *
import numpy as np
from dataGenerator import *
from mixtureClassifier import *
import multiprocessing as mp


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
        self.prop_source = prop_source
        self.prop_target = prop_target

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
        self.output = dict()
        self.workers = mp.cpu_count()
        self.output['test-data'] = dict()
        self.output['test-data']['bayes-error'] = np.mean((s['y']-s['bayes'])**2)


    def _QLabledClassifier(self):

        cl = WithLabelOptimalClassifier()
        cl.fit(x_source=self.data['source-data']['x'], y_source=self.data['source-data']['y'], x_target=self.data['target-data']['x'], y_target=self.data['target-data']['y'])
        self.output['labeled-data'] = dict()
        s = self.output['labeled-data']
        s['bandwidth'] = cl.bandwidth
        y_predict = cl.predict(self.data['test-data']['x'])
        s['error'] = np.mean((y_predict - self.data['test-data']['y'])**2)


    def _QUnlabeledClassifier(self):

        cl = WithoutLabelClassifier()
        cl.fit(x_source=self.data['source-data']['x'], y_source=self.data['source-data']['y'], x_target=self.data['target-data']['x'])
        y_predict = cl.predict(self.data['test-data']['x'])
        self.output['unlabeled-data'] = dict()
        s = self.output['unlabeled-data']
        s['bandwidth'] = cl.bandwidth
        s['error'] = np.mean((y_predict - self.data['test-data']['y'])**2)



    def _MixtureClassifier(self):
        cl = OptimalMixtureClassifier(nodes = self.workers)
        cl.fit(x_source=self.data['source-data']['x'], y_source=self.data['source-data']['y'], x_target=self.data['target-data']['x'], y_target=self.data['target-data']['y'])
        y_predict = cl.predict(self.data['test-data']['x'])
        self.output['mixture-classifier'] = dict()
        s = self.output['mixture-classifier']
        s['mixture'] = cl.mixture
        s['error'] = np.mean((y_predict - self.data['test-data']['y'])**2)


    def _ClassicalClassifier(self):
        cl = KDEClassifierOptimalParameter()
        cl.fit(x = self.data['target-data']['x'], y = self.data['target-data']['y'])
        y_predict = cl.predict(self.data['test-data']['x'])
        self.output['classical-classifier'] = dict()
        s = self.output['classical-classifier']
        s['bandwidth'] = cl.bandwidth
        s['error'] = np.mean((y_predict - self.data['test-data']['y'])**2)


    def _OracleClassifierNoTargetLabel(self):
        cl = KDEClassifierOptimalParameter()
        cl.fit(x = self.data['source-data']['x'], y = self.data['source-data']['y'])
        bandwidth = cl.bandwidth
        cl = KDEClassifier(bandwidth)
        w = np.array([(1-self.prop_target)/(1-self.prop_source), self.prop_target/self.prop_target])
        cl.fit(X = self.data['source-data']['x'], y = self.data['source-data']['y'], weights = w)
        y_predict = cl.predict(self.data['test-data']['x'])
        self.output['oracle-classifier'] = dict()
        s = self.output['oracle-classifier']
        s['bandwidth'] = cl.bandwidth
        s['error'] = np.mean((y_predict - self.data['test-data']['y'])**2)

       

