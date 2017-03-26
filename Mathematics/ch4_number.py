# coding: utf-8
# chapter 4: number theory


from numpy import *
#import numpy as np

#-------------------------
# 4.1: divisibility
#-------------------------

def gcd(m,n): # 0<=m<n
	if m==0:
		return n
	else:
		return gcd(n%m, m)

def getApostrophe(m,n):
	# apostrophe　n. 书写中撇号(')(表示省略或所有格)
	# 读作prime，有时也读成primed。
	# a……是词汇，类比中文“撇号”　　prime是正确读法
	if m==0:
		mp = 0;	np=1
		#return mp,np
	else:
		r = n%m
		rbar,mbar = getApostrophe(r,m)
		mp = mbar-floor(n/m)*rbar
		np = rbar
	return mp,np

def lcm(m,n):
	p = gcd(m,n)
	return m*n/p # (m/p)*(n/p)*p
	# gcd(m,n) * lcm(m,n) = m*n

'''
print gcd(12,18)
mp,np = getApostrophe(12,18)
print mp,np
print lcm(12,18)
'''


#-------------------------
# 4.5 relative primality
#-------------------------

I = mat([[1,0],[0,1]])
L = mat([[1,1],[0,1]])
R = mat([[1,0],[1,1]])

# S is a string, eg, 'LRRL'
# M(S) is a matrix, i.e., mat([[n,np],[m,mp]])

def ms(S):
	ls = len(S)
	ans = I
	for i in range(ls):
		if S[i]=='L':
			ans = ans*L
		elif S[i]=='R':
			ans = ans*R
	return ans

def fs(S):
	M = ms(S)
	n = M[0,0];	np = M[0,1]
	m = M[1,0];	mp = M[1,1]
	return str(m+mp)+'/'+str(n+np)

#print ms('LRRL')
#print fs('LRRL')

def fsFloat(MS): # matrix
	n = MS[0,0];	np = MS[0,1]
	m = MS[1,0];	mp = MS[1,1]
	return float(m+mp)/float(n+np)

def getLocation(m,n):
	S = I
	print 'f('+str(m)+'/'+str(n)+') = ',
	while abs(float(m)/float(n) - fsFloat(S)) >= 1e-6:
		if float(m)/float(n) < fsFloat(S):
			print 'L',
			S = S*L
		else:
			print 'R',
			S = S*R
	print '\n',

def getPosition(m,n):
	print 'f('+str(m)+'/'+str(n)+') = ',
	while m != n:
		if m<n:
			print 'L',
			n = n - m
		else:
			print 'R',
			m = m - n
	print '\n',

def irrational(alpha): # present alpha,  an irrational number
	print 'alpha =',alpha,'=',
	iteration = 20
	while iteration>=0: # True, run (iteration+1) times
		if alpha<1:
			print 'L',
			alpha = alpha/(1.0-alpha)
		else:
			print 'R',
			alpha = alpha-1
		iteration -= 1
	print '\n',

'''
print fsFloat( mat([[3,4],[2,3]]) )
print fsFloat(I),fsFloat(L),fsFloat(R)
getLocation(5,7)
getPosition(5,7)
irrational(2.71828) # e
'''


#-----------------------------
# 4.6 'mod', the congruence relation
# 4.7 independent residues
#-----------------------------

def congruence(m,n, x,y): # compute by 同余式
	# (1,0)=a=n'*n 	(0,1)=b=m'*m
	mp,np = getApostrophe(m,n)
	a = np*n;	b = mp*m
	if mp<0:
		b = b%(m*n)
	if np<0:
		a = a%(m*n)
	ans = (a*x + b*y)%(m*n)
	#print '(x mod '+str(m)+', x mod '+str(n)+')'
	#print '(1,0) =',a,'\t(0,1) =',b
	print '('+str(x)+','+str(y)+') =',int(ans)
	return ans

'''
m = 3;	n = 5
mp, np = getApostrophe(m,n)
print mp,np
congruence(m,n, 2,3)

for k in range(m*n):
	x = k%m
	y = k%n
	congruence(m,n, x,y)
'''



