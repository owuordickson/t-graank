
import numpy as np

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

print(tempp)
print()
print(tempm)