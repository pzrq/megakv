#!/usr/bin/python
import os
import time
import shutil
import sys

#for stream in range(1,9):
for stream in [1,2,4,6,8]:
	print '-----------------' + str(stream)
	#for thread in range(4096,33792,4096):
	for thread in [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]:
		#print thread
		#for threads_per_blk in range(256, 1152, 256):
		for threads_per_blk in [64, 128, 256, 512, 1024]:
			cmd = './run ' + str(stream)+ ' ' + str(thread) + ' ' + str(threads_per_blk)
			#print cmd
			os.system(cmd)
		print ''
