#!/usr/bin/python
import os
import time
import shutil
import sys

#for job_num in range(15000, 51000, 5000):
#for job_num in range(12000, 41000, 4000):
#for job_num in range(9000, 31000, 3000):
#for job_num in range(1200, 4100, 400):
#for job_num in range(1500, 5100, 500):
#for job_num in range(60000, 65000, 10000):
#for job_num in range(24000, 81000, 8000):
#	for stream in range(1, 9, 1):
#for job_num in [500]:
#for job_num in range(19000, 380100, 19000):
#for job_num in range(900, 3300, 300):
#for job_num in range(1000, 20100, 1000):
#for job_num in range(10000, 105000, 10000):
for job_num in range(1000, 10100, 1000):
	for stream in [1, 6]:
		cmd = './run ' + str(stream)+ ' ' + str(job_num)
		#print cmd
		os.system(cmd)
	print ''
