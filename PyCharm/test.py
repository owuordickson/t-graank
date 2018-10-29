from PyCharm.algorithm.TimeLag import *

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

a = [1,2,3,4,5]
mode = stats.mode(a)
print(mode)