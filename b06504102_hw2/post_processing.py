import csv
import numpy as np
import torch

with open('pred_41.csv','r') as fp:
    Y = list(csv.reader(fp))
    Y = np.array(Y[1:])[:,1:]

Y = Y.flatten()

count=0
for i in range(len(Y)):
    if i == 0 or i == len(Y)-1:
        continue
    if (Y[i-1] == Y[i+1]) and (Y[i] != Y[i-1]):
        count+=1
        Y[i] = Y[i-1]

print(count/451552.)

with open('pred_41_ABA.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(Y):
        f.write('{},{}\n'.format(i, y))
