import cProfile
import os

def jobs():
    m, n, n_test, d, prop, distance, index = 1000, 100, 1000, 3, 0.8, 0.8, 1
    os.system(f'python3 experiment2.py {m} {n} {n_test} {d} {prop} {distance} {index}')
 

cProfile.run('jobs()')
