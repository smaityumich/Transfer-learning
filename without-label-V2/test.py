import multiprocessing as mp
import numpy as np


def f(par):
    x, y = par 
    print(f'{x} and {y} multiplies to {x*y}\n')

x = [1,2,3,4,5]
y = [1,2,3,4,5]
par_list = np.meshgrid(x,y)
par_list = np.array(par_list)
par_list = par_list.reshape((2,-1))
par_list = par_list.T
par_list = [tuple(par) for par in par_list]

pool = mp.Pool(mp.cpu_count())
pool.map(f, par_list)
pool.close()
