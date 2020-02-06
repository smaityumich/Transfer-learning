import numpy as np
import multiprocessing as mp
import os

m_vec = [25, 50, 100, 200, 400, 800, 1600, 3200]
n_vec = [500]
n_test_vec = [1000]
d = [5]
prop = [0.8]
distance = [0.5]
iter_vec = range(100)
par_list = np.meshgrid(m_vec, n_vec, n_test_vec, d, prop, distance, iter_vec)
par_list = np.array(par_list)
par_list = par_list.reshape((7, -1))
par_list = par_list.T
par_list = [tuple(par) for par in par_list]



outdir = os.getcwd() + '/out'
if not os.path.exists(outdir):
    os.system(f'mkdir {outdir}')






def jobs(par):
    m, n, n_test, d, prop, distance, index = par
    os.system(f'python3 experiment2.py {m} {n} {n_test} {d} {prop} {distance} {index}')
   

pool = mp.Pool(mp.cpu_count())
pool.map(jobs, par_list)
pool.close()
