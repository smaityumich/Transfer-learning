import json
import os

list_dict = []

for filename in os.listdir('.result/'):
    with open('.result/'+filename, 'r') as fh:
        d = fh.read()
        list_dict.append(d)
#        print(d+'\n')

with open('result.json', 'w') as f:
    for s in list_dict:
        f.writelines(s+'\n')



