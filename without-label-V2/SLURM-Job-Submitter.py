
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



job_file = 'submit.sbat'

# Experiment 1
n_sources = np.array(range(1, 33))*50


for n_source in n_sources:
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=n_P-{n_source}.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=10\n')
        fh.writelines('#SBATCH --mem-per-cpu=6gb\n')
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --account=stats_dept1\n")
        fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 unit_experiment.py {n_source} 200 2500 0.5 0.8 1 5 100")

    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')


# Experiment 2
n_targets = np.array(range(1, 33))*50


for n_target in n_targets:
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=n_Q-{n_target}.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=10\n')
        fh.writelines('#SBATCH --mem-per-cpu=6gb\n')
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --account=stats_dept1\n")
        fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 unit_experiment.py 200 {n_target} 2500 0.5 0.8 1 5 100")

    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
