# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:07:52 2019

@author: Subha Maity
"""

import numpy as np
import scipy.stats as st
from pyspark import SparkConf, SparkContext
from AdaClassifier import *

if len(sys.argv) != 2:
  print('Usage: ' + sys.argv[0] + ' <out>')
  sys.exit(1)
outputloc = sys.argv[1]

conf = SparkConf().setAppName('Summation')
sc = SparkContext(conf=conf)



def vec_sum(x,y):
    return [(x[i]+y[i])/100 for i in range(len(x))]

        
        

keys = sc.parallelize(par)
data = keys.map(get_data)
data = data.flatMap(lambda x: x)
error = data.map(classify)
generalized_error = error.reduceByKey(vec_sum)
data.saveAsTextFile(outputloc)
sc.stop()

