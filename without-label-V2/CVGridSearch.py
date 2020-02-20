from sklearn.model_selection import KFold, ParameterGrid
from multiprocessing import Pool
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import base

class CVGridSearch(BaseEstimator):

    def __init__(self, method, param_dict, cv = 5, nodes = 1):
        self.param_dict = param_dict
        self.params = list(ParameterGrid(self.param_dict))
        self.params = [str(arg) for arg in self.params]
        self.methods = [base.clone(method).set_params(**eval(arg_str)) for arg_str in self.params]
        self.kf = KFold(n_splits=cv)
        self.cv = cv
        self.workers = nodes


    def unit_work(self, args):
        method, arg_str, data = args
        x_source, y_source, x_target, y_target = data
        errors = np.zeros((self.cv,))
        
        for index, (train_index, test_index) in enumerate(self.kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            method.source_data(x_source, y_source)
            method.fit(x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_pred - y_test)**2)

        return {'args': eval(arg_str), 'error-cv': errors, 'error': np.mean(errors)}

    def fit(self, x_source, y_source, x_target, y_target):
        data = x_source, y_source, x_target, y_target
        datas = [data for _ in range(len(self.params))]
        with Pool(self.workers) as pool:
            list_errors = pool.map(self.unit_work, zip(self.methods, self.params, datas))
        self.evaluation = list_errors
        error_list = np.array([s['error'] for s in list_errors])
        self.best_param_ = eval(self.params[np.argmin(error_list)])




