import os
import numpy as np

job_file = 'submit.sbat'


def expt(ns, nt, ntest, ps, pt, dist, d):

    iteration = 20
    for i in range(iteration):
        os.system(f'touch {job_file}')

        
        with open(job_file,'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name=P{ns}-Q{nt}.job\n")
            fh.writelines('#SBATCH --nodes=1\n')
            fh.writelines('#SBATCH --cpus-per-task=1\n')
            fh.writelines('#SBATCH --mem-per-cpu=1gb\n')
            fh.writelines("#SBATCH --time=00:30:00\n")
            fh.writelines("#SBATCH --account=yuekai1\n")
            fh.writelines("#SBATCH --mail-type=NONE\n")
            fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
            fh.writelines('#SBATCH --partition=standard\n')
            fh.writelines(f"python3 unit_exptV3.py {ns} {nt} {ntest} {ps} {pt} {dist} {d} {i}")

        os.system("sbatch %s" %job_file)
        os.system(f'rm {job_file}')

#for ns in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
#    expt(ns, 100, 100, 0.5, 0.9, 0.4, 4)
for nt in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]:
    expt(100, nt, 100, 0.5, 0.9, 0.4, 4)

#for n in [25, 50, 100, 200, 400, 800, 1600]:
#    expt(2*n, n, 100, 0.5, 0.8, 1, 5)

#for ns in [100, 300, 700, 1500, 3100, 6300, 12700]:
#    expt(ns, 100, 100, 0.5, 0.9, 0.4, 4)
#
