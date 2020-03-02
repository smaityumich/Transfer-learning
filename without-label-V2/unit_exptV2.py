import os
import sys


if len(sys.argv) != 9:
    raise TypeError('Wrong input.')
    sys.exit(1)



ns, nt, ntest, ps, pt, dist, d, it = int(float(sys.argv[1])), int(float(sys.argv[2])), int(float(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(float(sys.argv[7])), int(float(sys.argv[8]))


os.system(f'python3 DataCreator.py {ns} {nt} {ntest} {ps} {pt} {dist} {d} {it}')

experiments = ['QLabeled', 'QUnlabeled', 'Mixture', 'Classical', 'Oracle']

for e in experiments:
    os.system(f'python3 experimentsV2.py  {ns} {nt} {ntest} {ps} {pt} {dist} {d} {it} {e}')

