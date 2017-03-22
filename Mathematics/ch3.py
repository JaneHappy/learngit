# chapter 3: 轮盘赌，中国老板的店

import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt
#import pylab as pl

'''
PART 1: ----------------------------------
'''
'''
def plotN(a,b):
    x = np.arange(float(a), float(b)+1, 1) #float(a), b+1,1
    y = pow(x, 1.0/3)
    z1 = floor(y)**2 *3
    z2 = floor(y) *15
    z3 = floor(x/floor(y)) *6
    z = z3 + z1 + z2 - 18
    #fig = plt.figure()
    #fig.clf()
    #ax = plt.subplot(111)
    #ax.plot(x,y, c='b', )
    plt.plot(x,x-y, c='g')
    #plt.plot(x,y, c='b')
    #plt.plot(x,x, c='r')
    #plt.xlabel('N');    plt.ylabel('some numbers'); #plt.title('')
    plt.show()

p = pow(x[-1],1.0/3)
print "%.10f\n%.10f\n" % (p, math.floor(p) )
print p<10.0, abs(p-10) <10**(-3)
'''
# print pow(415.0/27, 1.0/3)
'''
a = 1
b = 1250 # 10**3, 10**4, 5000, 2000, 
x = arange(float(a), float(b)+1, 1)
y = pow(x, 1.0/3) #>=1
y = floor(y)
z = ceil( pow(float(b)+1, 1.0/3) ) + 1
for i in range(int(z)):
    idx = (x == pow(float(i),3.0))
    y[idx] = i
w = floor(x/y) + y**2/2.0 + y*5.0/2.0 - 3.0 
plt.plot(x, 6.0*w-x, 'b')
plt.plot(x, np.zeros(x.shape), 'r')
#pl.ylim()
plt.show()
'''
'''
print y2[-1]
for i in range(len(x)):
    if y[i]<1:
        if abs(y[i]-1) < 10**(-6):
            z[i] = 1.0
    elif y[i]<2:
        if abs(y[i]-2) < 10**(-6):
            z[i] = 2.0
    elif y[i] < 3:
        if abs(y[i]-3) < 10**(-6):
            z[i] = 
'''





'''
# PART 1
#a = 1064 # 1000,  998, 727, 1000, 
#b = 1082 # 1250, 1002, 731, 1100,
# PART 2
a = 637 #    1, 600, 630, 
b = 657 # 1100, 700, 660, 
pa = floor( float(a-1)**(1.0/3) ) - 1
pb = ceil( float(b+1)**(1.0/3) ) + 1
for n in arange(float(a), float(b)+1, 1):
    k = n**(1.0/3)
    for i in arange(int(pa), int(pb)+1, 1):
        if n == i**3:
            k = float(i)
    k = floor(k)
    w = floor(n/k) + k*(k+5)/2 - 3
    # PART 1
    #print 'N =',n, '\tK =',k, '\tW =',w, '\twin =',int(6*w-n), '\tWinPr =',(6*w-n)/float(n)
    #plt.plot(n, 6*w-n, 'b.')
    # PART 2
    print 'N =',n, '\tK =',k, '\tW =',w, '\twin =',int(5*w-n), '\tWinPR = ',(5*w-n)/float(n)
    plt.plot(n, 5*w-n, 'b.')
plt.plot([a,b], [0,0], 'r')
plt.show()
'''



# PART 3, PART 4

def subtriplicate(x):
    t = x**(1.0/3)
    xa = floor(t)-2
    xb = ceil(t)+2
    for i in arange(xa, xb, 1):
        if x == i**3:
            t = float(i)
    return t
#print floor(subtriplicate(728)), 1000,729

def winTimes(m,n):
    km = floor(subtriplicate(m))
    kn = floor(subtriplicate(n))
    w = floor(float(n)/kn) + ((kn*km-km-1)*3 + kn*(kn+2))/2
    return w,km,kn

# M = [1,1000], N = [1000,1000]
# part 3
M = [1,75] #[1,1000],  [1,200], [25,75]
N = [1000,1000] #[1000,1000]

for m in arange(M[0], M[1]+1, 1):
    for n in arange(N[0], N[1]+1, 1):
        w, km,kn = winTimes(m,n)
        #wt = 6*w-n #part3
        wt = 5*w-n #part4
        print 'm=',m, '\tN=',n, '\tw=',w, '\tkm=',km,'kn=',kn, 
        print '\twin=',wt, 'winPr=',wt/float(n)
        # part 3
        #print '\twin=',(6*w-n), 'winPr=',(6*w-n)/float(n)
        # part 4
        #print '\twin=',(5*w-n), 'winPr=',(5*w-n)/float(n)
        plt.plot(m, wt, 'b.')
plt.plot(M, [0,0], 'r')
plt.show()










