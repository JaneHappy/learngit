#!/usr/bin/env python
# coding=utf-8


# chapter 4


import numpy as np
import math


def findPrime(x): #埃拉托斯提尼斯筛
	numbers = np.arange(0,x+1,1) # 2~x
	idx = np.ones((1,len(numbers))) * (-1)
	items = len(numbers) - 2 # -0,-1 #=x-1
	
	for i in range(2, items+1, 1):
		if idx[0][i] == -1:
			idx[0,i] = 1 # prime
			for k in range(2, items+1, 1):
				if k*numbers[i] <= x:
					idx[0][k*numbers[i]] = 0

	#print "x =",x, "Pi(x) =", sum(idx[0]==1), "NotPrime:", sum(idx[0]==0), "All:",items, "\nPrimes:", #idx[0].count(0)
	prime = []
	for i in range(2, items+1, 1):
		if idx[0][i] == 1:
			prime.append(numbers[i])
			#print numbers[i],

	return prime #numbers,idx,


def eculidNumber(x):
	idx,m,n = 1,2,3 # e1=1, e2=3,e3=7, ... e_m,e_n, m<n
	while idx<=x:
		yield m
		m,n = n, (n-1)*n+1
		idx = idx + 1


def isPrime(x):
	if x == 1:
		return False
	elif x < 1:
		print "Wrong Number Error: x must >=2"
		return False
	else:
		pass

	maxNum = math.ceil(math.sqrt(x)) #math.sqrt(x), math.pow(x,1.0/2), x**(1.0/2)
	for i in range(2, int(maxNum), 1):
		if x % i == 0:
			return False
	return True

def mersenneNumber(x):
	prime = findPrime(2**x)
	mersenne = []
	for p in prime:
		t = int(2**p-1)
		if isPrime(t) == True:
			mersenne.append(5)
			print "p=",p, '\tMersenneNumber:',(2**p-1)
	return mersenne



#prime = findPrime(10)
#print isinstance(prime[0], float), isinstance(prime[0], int)  #False,True

#for e in eculidNumber(10):
#	print e

#print isPrime(10), isPrime(1), isPrime(7)
mersenneNumber(4) #5