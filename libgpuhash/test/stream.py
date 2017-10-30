#!/usr/bin/python
import os
import time
import shutil

for s in range(1,9):
	cmd = './run ' + str(s)
	os.system(cmd)
