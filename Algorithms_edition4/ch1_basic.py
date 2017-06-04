# coding: utf-8
# chapter 1: Basic



import numpy as np 

# 欧几里得算法，两个非负整数
def gcd(p,q):
	# non-negetive integer (p,q>=0)
	if q==0:
		return p
	r = p%q
	return gcd(q,r)
#print gcd(18,12) #gcd(18,30)



# chap 1.1.10  二分查找
class BinarySearch(object):
	def __init__(self):
		pass
	def rank(self, key, a):
		low = 0
		high = len(a) - 1
		while low<=high:
			mid = low + (high - low)/2
			if key<a[mid]:
				high = mid - 1
			elif key>a[mid]:
				low = mid + 1
			else:
				return mid
		return -1
	def main(self, a): # int a
		whitelist = a
		a = sorted(a)
		iteration = 7
		while iteration>=0:
			key = raw_input("Please input: ") # 3.x: input("Enter your name: ")
			key = int(key)
			if self.rank(key, a)<0:
				print key, 'is not in the list.'
			iteration -= 1
		print 'The list is:', a

#test = BinarySearch()
#test.main([9,3,8,2,1,0,11])
'''
Pleas input: 9
Pleas input: 3
Pleas input: 94
94 is not in the list.
Pleas input: 135
135 is not in the list.
Pleas input: 35
35 is not in the list.
Pleas input: 4
4 is not in the list.
Pleas input: 5
5 is not in the list.
Pleas input: 142
142 is not in the list.
The list is: [0, 1, 2, 3, 8, 9, 11]
'''




