# coding: utf-8
# http://blog.csdn.net/lanbing510/article/details/8487997



import sys
import time

print 'Dive in'
saveout = sys.stdout
fsock = open('out.log', 'w')
sys.stdout = fsock
print 'This message will be logged instead of displayed.'
print 'Finished at %.6f s.' % time.clock()
sys.stdout = saveout
fsock.close()

'''
Dive in
[Finished in 0.2s]
'''