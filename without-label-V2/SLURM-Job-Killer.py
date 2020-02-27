import re
import os

os.system('squeue -u smaity > jobs.list')
with open('jobs.list') as f:
    out = f.read()

x = re.compile(r'\s\d{2,}\s')
l = re.findall(x, out)

for job_id in l:
    os.system(f'scancel {job_id}')

os.system('rm jobs.list')
