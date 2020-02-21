
import os
import numpy as np

def mkdir_path(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.system(f'mkdir {dir}')
        
cwd = os.getcwd() + '/'
output_dir = cwd + '.out'

mkdir_path(output_dir)
log_dir = cwd + '.log'
mkdir_path(log_dir)

# Experiment 1
n_sources = [50, 100, 200, 400, 800, 1600, 3200, 6400]



job_file = 'submit.sbat'


for n_source in n_sources:
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=n_source-{n_source}.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=6gb\n')
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --account=stats_dept1\n")
        fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 unit_experiment.py {n_source} 200 2500 0.5 0.8 0.8 5 100")

    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
    os.system(f'mv *.out {log_dir}') 
