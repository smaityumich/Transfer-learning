import numpy as np


class DataGenerator():
    
    def __init__(self, d = 4):
        self.d = d
        
    def _generateY(self, n = 100, prop = 0.5):
        self.prop = prop
        self.n = n
        self.y = np.random.binomial(1, self.prop, (self.n,))
        
    def _generateX(self, distance = 0):
        self.mu = distance
        f = lambda y : np.random.normal(loc = self.mu, scale = 1, size = (self.d,))  if y else  np.random.normal(loc = 0, scale = 1, size = (self.d,)) ## Generates data from N_d(mu, I_d) if label=1, else from N_d(0,I_d) if label=0
        self.x = [f(y) for y in self.y]
        
    def _bayesDecision(self, x):
        x = np.array(x)
        prior = np.log(self.prop/(1-self.prop))
        log_lik_ratio = 0.5*np.sum(x**2) - 0.5*np.sum((x-self.mu)**2)  ## Calculates log-likelihood ratio for normal model Y=1: N(mu, 1); Y=0: N(0,1)
        posterior = prior + log_lik_ratio
        return 0 if posterior<0 else 1
        
    def _bayesY(self):
        self.bayesLabel = [self._bayesDecision(x) for x in self.x]
        
    def _getData(self, n = 100, prop = 0.5, distance = 0.4):
        self._generateY(n, prop)
        self._generateX(distance)
        self._bayesY()
        return np.array(self.x), np.array(self.y), np.array(self.bayesLabel)


        
        
 
