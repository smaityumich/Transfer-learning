# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:01:42 2020

@author: Subha Maity
"""

import numpy as np

class GeneratorClassification():
    
    def __init__(self, d):
        
        ## Debugging faulty dimension input
        try: 
            d = int(d)
        except: 
            raise TypeError('d can\'t be converted into integer.')