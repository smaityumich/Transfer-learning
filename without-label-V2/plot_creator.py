import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.ndimage.filters import gaussian_filter1d


filename = sys.argv[1]

with open(filename, 'r') as f:
    result = f.read()

res = re.split(r'\n', result)
res.pop()
result_dict = dict()

for index, string in enumerate(res):
    result_dict[index] = eval(string)

df = pd.DataFrame(result_dict).T
df = df.astype({'n-source': 'int32', 'n-target': 'int32', 'error': 'float32', 'expt': 'str'})
summary = df.groupby(['n-source', 'expt'])['error'].mean().to_frame()

fig, ax = plt.subplot(figsize = (4, 4))
col = {'Classical_bandwidth': ['b', 'Classical'], 'QLabeled': ['r', 'QLabeled'], 'Mixture_bandwidth': ['g', 'Mixture']}
ns = [200, 400, 800, 1600, 3200, 6400, 12800]
bayes = 0.08

for key, grp in summary.groupby(['expt']):
    y = grp.to_numpy()
    y = np.reshape(y, (-1,))
    smooth = gaussian_filter1d(y - bayes, sigma = 0.8)
    ax.plot(ns, smooth, label = col[key][1], c = col[key][0])

ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(loc = 'best')
plt.savefig('plot-np-test.pdf')

