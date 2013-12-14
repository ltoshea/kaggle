#!/usr/bin/python
import csv
import sys
import numpy as np
import pdb

def fread(f=None,train=False):
	"""Reads in test and training CSVs. Training has label in col1"""
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


def fwrite(f=None,result):
	if (f==None):
		print("No filename given for output file. Exiting...")
		sys.exit(1)
	f = open("output.knn-kdtree.csv", "w")
	for r in result:
	    f.write(str(r))
	    f.write("\n")
	f.close()

if __name__ == "__main__":
	train,labels = filein(f='data/train.csv', train=True)
	test, tmplbl = filein(f='data/test.csv')