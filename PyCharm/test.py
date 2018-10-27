
import numpy as np
import skfuzzy as fuzzy
from PyCharm.TimeLag import *

#x = [True,False,False]
#print(x)
#y = np.sum(x)
#print(y)

n=5
tempp = np.zeros((n, n), dtype='bool')
tempm = np.zeros((n, n), dtype='bool')

for j in range(n):
    for k in range(j + 1, n):
        tempp[j][k] = 1
        tempm[k][j] = 1

x,y = optimize_timelag(0.5,[4001],[4000,4567,5000],[2000,5000])
print(x)
print(y)