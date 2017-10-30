#!/usr/bin/python
import os
import time
import shutil
import sys

for thread_num in range(9216, 33792, 1024):
	print '-------------' + str(thread_num)
	for job_num in range(18000, 61000, 6000):
		for stream in range(1,9):
			cmd = './run ' + str(stream)+ ' ' + str(job_num) + ' ' + str(thread_num)
			#print cmd
			os.system(cmd)
		print ''
