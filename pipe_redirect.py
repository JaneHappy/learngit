# coding: utf-8
# http://blog.csdn.net/lanbing510/article/details/8487997


'''
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

'''
Dive in
[Finished in 0.2s]

out.log------------
This message will be logged instead of displayed.
Finished at 0.000003 s.
'''





# https://gist.github.com/hxer/810f791a49e4e1569637
#将日志同时输出到文件和屏幕
import logging
logging.basicConfig(level=logging.DEBUG, 
	format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', 
	datefmt='%a, %d %b %Y %H:%M:%S',
	filename='pipe_out.log',#myapp.log',
	filemode='w')

#################################################################################################
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
#################################################################################################

logging.debug('This is debug message')
logging.info('This is info message')
logging.warning('This is warning message')

'''
root        : INFO     This is info message
root        : WARNING  This is warning message
[Finished in 0.2s]
'''


