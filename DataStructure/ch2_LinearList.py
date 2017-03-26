# coding: utf-8

# data structure
# chapter 2: linear list


from numpy import *
import numpy as np

#prng = np.random.RandomState(123456789) # define a local seed

# eg 2-1
la = np.random.random(size=20)
lb = np.random.random(size=10)
def union(la,lb): # insert all elements in lb but not in la into la
	# obtain lengths of linear lists
	len_la = len(la)
	len_lb = len(lb)
	lc = np.zeros(shape=(1,len_la+len_lb))
	lc[0:len_la] = la
	len_lc = la
	for i in range(len_lb):
		e = lb[i]
		flag = False
		for j in range(len_la):
			if e==la[j]:
				flag = True
				break
		if flag == False:
			#la.insert(e)
			lc[len_la+i] = e
			len_lc += 1
	lc = lc[0:len_lc]
	print lc

print "la =",la,"\nlb =",lb
union(la,lb)
