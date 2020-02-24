import numpy as np
import functools

class DataGenerator():
    
    def __init__(self, d = 4):
        self.d = d
        
    def _generateY(self, n = 100, prop = 0.5):
        self.prop = prop
        self.n = n
        self.y = np.random.binomial(1, self.prop, (self.n,))

    def _generate_unitX(self, y):
        return np.random.triangular(0, 0.5, 1, (self.d,)) if y else np.random.random((self.d,)) 

        
    def _generateX(self, distance = 0):
        self.mu = distance/np.sqrt(self.d)
        self.x = [self._generate_unitX(y) for y in self.y]

    def _density1(self, x):
        f = lambda u: 4*u if u < 0.5 else 4*(1-u)
        density = map(f, x)
        return functools.reduce(lambda u, v : u*v, density, 1)
        
    def _bayesDecision(self, x):
        x = np.array(x)
        prior = np.log(self.prop/(1-self.prop))
        log_lik_ratio = np.log(self._density1(x)) ## Calculates log-likelihood ratio for triangular vs uniform in [0,1]^d
        posterior = prior + log_lik_ratio
        return 0 if posterior<0 else 1
        
    def _bayesY(self):
        self.bayesLabel = [self._bayesDecision(x) for x in self.x]
        
    def _getData(self, n = 100, prop = 0.5, distance = 0.4):
        self._generateY(n, prop)
        self._generateX(distance)
        self._bayesY()
        return np.array(self.x), np.array(self.y), np.array(self.bayesLabel)


        
        
 
