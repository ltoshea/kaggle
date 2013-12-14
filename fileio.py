#!/usr/bin/python
import csv
import sys
import numpy as np
import pdb

def fread(f=None,train=False):
	"""Reads in test and training CSVs. (note:training has label in col1)"""
	digits = []
	labels = []

	if (f==None):
		print("No file given to read, exiting...")
		sys.exit(1)

	digiread = csv.reader(open(f,'r'),delimiter = ',')
	startrow = 0;
	for row in digiread:
		startrow +=1
		if (startrow==1):
			continue #Stops Value error
		if (train == True):
			labels.append(int(row[0]))
			row = row[1:]
		digits.append(np.array(np.int64(row)))

	"""for i in range (0,1):
			#pdb.set_trace()
			#print ("labels",labels[i])
			print("digits:",digits[i])"""
	return(digits,labels)


def fwrite(result,fname=None):
	"""Output to file"""
	if (fname==None):
		print("No filename given for output file. Exiting...")
		sys.exit(1)
	fname = open(fname, "w")
	for r in result:
		#print "type r: ",type(r)
		fname.write(str(r))
		fname.write("\n")
	fname.close()

if __name__ == '__main__':
	train,labels = fread(f='data/train.csv', train=True)
	test, tmplbl = fread(f='data/test.csv')