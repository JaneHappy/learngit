# coding: utf-8

# data structure
# chapter 2: linear list


from numpy import *
import numpy as np


prng = np.random.RandomState(123456789) # define a local seed


# eg 2-1
la = np.random.random(size=20) #20
lb = np.random.random(size=10) #10
la = floor(la*100)
lb = ceil(lb*100)

def union(la,lb): # insert all elements in lb but not in la into la
	# obtain lengths of linear lists
	len_la = len(la)
	len_lb = len(lb)
	lc = np.zeros(shape=(1,len_la+len_lb))
	lc = lc[0]
	lc[0:len_la] = la
	len_lc = len_la
	for i in range(len_lb):
		e = lb[i]
		flag = False
		for j in range(len_la):
			if abs(e-la[j])<1e-6: #e==la[j]:
				flag = True
				break
		if flag == False:
			#la.insert(e)
			lc[len_lc] = e
			len_lc += 1
	lc = lc[0:len_lc]
	print "union(la,lb) =",lc

#print "la =",la,"\nlb =",lb
#union(la,lb)


# eg 2-2
la = [3,5,8,11]
lb = [2,6,8,9,11,15,20]
lc = []
def mergeList(la,lb,lc):
	# in non-decreasing order
	sa = len(la);	sb = len(lb)
	i = j = 0;		k = 0
	#lc = []
	while (i<sa) and (j<lb): # not empty
		ai = la[i];	bj = lb[j]
		if ai<=bj:
			lc.append(ai);	k+=1;	i+=1
		else:
			lc.append(bj);	k+=1;	j+=1
	while (i<sa):
		ai = la[i];		i+=1
		lc.append(ai);	k+=1
	while (j<sb):
		bj = lb[j];		j+=1
		lc.append(bj);	k+=1
	print "MergedList = ",lc

#print 'la =',la,'\nlb =',lb
#mergeList(la,lb,lc) #mergeList(la,lb)
#print "lc =",lc


# alg 2.6
'''
la = np.random.random(size=7)
lb = np.random.random(size=4)
la = round(la*100);		lb = round(lb*100)

def mergeList_sq(la,lb): # in non-decreasing order
	lc = []
'''


